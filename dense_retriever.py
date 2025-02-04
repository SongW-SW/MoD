import json
import logging
import faiss
import hydra
import hydra.utils as hu
import numpy as np
import torch
import tqdm
import os
from transformers import set_seed
from torch.utils.data import DataLoader
from src.utils.dpp_map import fast_map_dpp, k_dpp_sampling
from src.utils.misc import parallel_run, partial
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.models.biencoder import BiEncoder
from scipy.cluster.vq import kmeans, vq
import random
from transformers import pipeline
from transformers import BertModel, BertTokenizer
import spacy
from spacy_transformers import Transformer
from collections import OrderedDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
# Initialize the pipeline with the task and model
from torch import nn

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, cfg) -> None:
        self.cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model_config = hu.instantiate(cfg.model_config)
        if cfg.pretrained_model_path is not None:
            self.model = BiEncoder.from_pretrained(cfg.pretrained_model_path, config=model_config)
        else:
            self.model = BiEncoder(model_config)
        self.cfg=cfg

        self.model = self.model.to(self.cuda_device)
        self.model.eval()

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader.dataset_split == "train"

        self.dpp_search = cfg.dpp_search
        self.dpp_topk = cfg.dpp_topk
        self.mode = cfg.mode
        # if os.path.exists(cfg.faiss_index):
        #     logger.info(f"Loading faiss index from {cfg.faiss_index}")
        #     self.index = faiss.read_index(cfg.faiss_index)
        # else:
        self.index, self.cluster_indexes = self.create_index(cfg)
        self.retrieved_data = json.load(open('./retrieved.json'))

    def create_index(self, cfg):
        dimension = 768  # 768  # Dimension of the embeddings


        logger.info("Building faiss index...")
        index_reader = hu.instantiate(cfg.index_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=index_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(index_reader, batch_size=cfg.batch_size, collate_fn=co)


        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension)) #768
        res_list = self.forward(dataloader, encode_ctx=True)

        self.train_text_list=[res['metadata']['text'] for res in res_list]


        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(embed_list, id_list)
        faiss.write_index(index, cfg.faiss_index)
        logger.info(f"Saving faiss index to {cfg.faiss_index}, size {len(index_reader)}")


        # Number of clusters
        n_clusters =15


        # Perform k-means clustering using scipy
        self.centroids, _ = kmeans(embed_list, n_clusters,iter=100,)
        # Assign each embedding to a cluster
        cluster_labels, distances_to_centroids = vq(embed_list, self.centroids)


        #distance_threshold = 0.3#0.3
        #within_threshold = distances_to_centroids < distance_threshold
        #embed_list = embed_list[within_threshold]
        #cluster_labels = cluster_labels[within_threshold]
        #id_list=id_list[within_threshold]

        #print(sum(within_threshold))


        np.savez('clustered_data.npz', embeddings=embed_list, labels=cluster_labels)

        # Create a dictionary of FAISS indexes, one for each cluster
        cluster_indexes = {}
        for cluster_id in range(n_clusters):
            # Initialize FAISS index for this cluster
            cluster_indexes[cluster_id] = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

        # Add embeddings to their respective cluster index
        for idx, cluster_id in enumerate(cluster_labels):
            cluster_indexes[cluster_id].add_with_ids(embed_list[idx:idx+1], id_list[idx:idx+1])

        print(cluster_indexes)

        # Save each cluster's index to a file (optional)
        for cluster_id, index_in_cluster in cluster_indexes.items():
            index_file = f"{cfg.faiss_index}_cluster_{cluster_id}.index"
            faiss.write_index(index_in_cluster, index_file)
            logger.info(f"Saving FAISS index for cluster {cluster_id} to {index_file}, size {index_in_cluster.ntotal}")


        return index, cluster_indexes


    def forward(self, dataloader, **kwargs):
        #nlp = spacy.load("en_core_web_trf")
        #nlp.to('cuda')

        #pos_pipeline = pipeline("token-classification", model="Jean-Baptiste/roberta-large-ner-english")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load pre-trained model
        model = BertModel.from_pretrained('bert-base-uncased').cuda()
        model.eval()

        tag_dict=OrderedDict([('ADJ', 0), ('ADP', 1), ('ADV', 2), ('AUX', 3), ('CCONJ', 4), ('DET', 5), ('INTJ', 6), ('NOUN', 7), ('NUM', 8), ('PART', 9), ('PRON', 10), ('PROPN', 11), ('PUNCT', 12), ('SCONJ', 13), ('SYM', 14), ('VERB', 15), ('X', 16)])

        linear_layer = torch.nn.Linear(in_features=768, out_features=768).cuda()
        res_list = []
        for i, entry in enumerate(tqdm.tqdm(dataloader)):

            #pos_tags = [pos_pipeline(one['text']) for one in entry['metadata']]
            #tag_sentence= [' '.join([tag['entity'] for tag in pos_tag]) for pos_tag in pos_tags]

            pos_tagging=False

            if pos_tagging:


                pos_tags=[nlp(one['text']) for one in entry['metadata']]
                tag_sentence = [' '.join([token.pos_ for token in pos_tag]) for pos_tag in pos_tags]

                #print([one['text'] for one in entry['metadata']])

                embs=[]
                for sentence in tag_sentence:
                    emb=[]
                    word_list=sentence.split(' ')
                    for key in tag_dict:
                        emb.append(word_list.count(key)/len(word_list))
                    embs.append(emb)
                embeddings=np.array(embs, dtype=np.float32)
                res=embeddings


            #encoded_input = tokenizer(tag_sentence, padding=True, truncation=True, return_tensors='pt')
            #encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}


            #with torch.no_grad():
            #    outputs = model(**encoded_input)
            #embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()


            with torch.no_grad():
                metadata = entry.pop("metadata")
                res = self.model.encode(**entry, **kwargs)
            res = linear_layer(res).cpu().detach().numpy()

            #res=np.concatenate((res,embeddings),axis=1)

            with torch.no_grad():

                res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])

        return res_list

    def find(self):
        print('################################')
        self.nli_model = AutoModelForSequenceClassification.from_pretrained('./mrpc_finetune').cuda()
        self.nli_tokenizer = AutoTokenizer.from_pretrained('./mrpc_finetune')



        res_list = self.forward(self.dataloader)
        for res in res_list:
            res['entry'] = self.dataset_reader.dataset_wrapper[res['metadata']['id']]



        if self.dpp_search:
            logger.info(f"Using scale_factor={self.model.scale_factor}; mode={self.mode}")
            func = partial(dpp, num_candidates=self.num_candidates, num_ice=self.num_ice,
                           mode=self.mode, dpp_topk=self.dpp_topk, scale_factor=self.model.scale_factor)
        else:
            if self.cfg.task_name=='mrpc':
                func = partial(knn_cluster_nli, num_candidates=self.num_candidates, num_ice=self.num_ice)
            else:
                func = partial(knn_cluster, num_candidates=self.num_candidates, num_ice=self.num_ice)


        set_global_object(self.index, self.is_train, self.cluster_indexes, self.nli_model, self.nli_tokenizer, self.train_text_list, self.retrieved_data)


        data = []
        for i,res in enumerate(tqdm.tqdm(res_list)):
            if self.cfg.task_name=='mrpc':
                data.append(func(res, retrieved_scores=self.retrieved_data[i]))
            else:
                data.append(func(res))

        #data = parallel_run(func=func, args_list=res_list, initializer=set_global_object,  n_processes=1,
        #                    initargs=(self.index, self.is_train, self.cluster_indexes, self.nli_model, self.nli_tokenizer, self.train_text_list))

        with open(self.output_file, "w") as f:
            json.dump(data, f)


def set_global_object(index, is_train, cluster_indexes, nli_model, nli_tokenizer, train_text_list, retrieved_data):
    global index_global, is_train_global, cluster_indexes_global, nli_model_global, nli_tokenizer_global, train_text_list_global, retrieved_data_global
    index_global = index
    is_train_global = is_train
    cluster_indexes_global = cluster_indexes
    nli_model_global = nli_model
    nli_tokenizer_global = nli_tokenizer
    train_text_list_global = train_text_list
    retrieved_data_global = retrieved_data


def find_top_indices(data, chunk_size=10, top_n=5):
    top_indices = []
    # Process each chunk
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        # Find indices of the top 5 elements in this chunk
        # np.argsort returns indices of sorted elements, [-top_n:] gets the last five (largest)
        indices = np.argsort(chunk)[:top_n]
        # Adjust indices to match their position in the original list
        adjusted_indices = [index + i for index in indices]
        top_indices.extend(adjusted_indices)

    top_indices_sorted_by_value = sorted(top_indices, key=lambda x: data[x])
    return top_indices_sorted_by_value

def classify_id(id, clusters):
    for i,cluster in enumerate(clusters):
        if id in cluster:
            return i

def knn_cluster_score(entry, retrieved_scores, num_candidates=10, num_ice=1):
    indices=[]
    for j in range(15):
        index_file = './output/dense/mrpc/EleutherAI/gpt-neo-2.7B/index_cluster_{}.index'.format(j)
        index=faiss.read_index(index_file)
        indices.append([index.id_map.at(i) for i in range(index.id_map.size())])

    retrieved_data=json.load(open('./output/epr/mrpc/EleutherAI/gpt-neo-2.7B/bert-fix_ctx-shared-bs64/train_retrieved.json'))[entry['metadata']['id']]

    clusters=[[index.id_map.at(i) for i in range(index.id_map.size())] for cluster_id, index in cluster_indexes_global.items()]

    selected_from_sets = {}
    all_near_ids=[]
    for example_id in retrieved_data['ctxs']:
        cluster_idx=classify_id(example_id, clusters)
        if cluster_idx not in selected_from_sets:
            selected_from_sets[cluster_idx]=example_id
            all_near_ids.append(example_id)
        if len(selected_from_sets)==len(clusters):
            break

    entry = entry['entry']
    entry['ctxs'] = all_near_ids[:num_ice]
    entry['ctxs_candidates'] = [[i] for i in all_near_ids[:num_candidates]]

    return entry

def knn_cluster_nli(entry, retrieved_scores, num_candidates=10, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    chunk_size=15
    top_n=chunk_size

    # Iterate over each cluster index and retrieve 10 nearest samples
    Is=[]
    scores_total=[]
    for cluster_id, index in cluster_indexes_global.items():

        ids=[index.id_map.at(i) for i in range(index.id_map.size())]
        scores=[retrieved_scores['scores_total'][i] for i in ids]

        D, I = index.search(embed, chunk_size)

        #selected_ids = np.argsort(-np.array(scores)).tolist()[:chunk_size]
        #ids = [ids[i] for i in selected_ids]
        #scores=[scores[i] for i in selected_ids]

        selected_ids=I[0].tolist()[:chunk_size]
        ids=selected_ids
        scores=[retrieved_scores['scores_total'][i] for i in ids]


        #Ds.extend((-scores)[ids].tolist())
        Is.extend(ids)
        scores_total.extend(scores)
        #Is.extend(ids)

    ids=find_top_indices(-np.array(scores_total),chunk_size=chunk_size, top_n=top_n)[:10]

    Ds=[-scores_total[i] for i in ids]
    all_near_ids=[Is[idx] for idx in ids]


    np.savez('selected_data.npz', data_ids=all_near_ids)
    #all_near_ids = random.sample(all_near_ids, len(all_near_ids))

    entry = entry['entry']
    entry['ctxs'] = all_near_ids[:num_ice]
    entry['ctxs_candidates'] = [[i] for i in all_near_ids[:num_candidates]]

    return entry

def knn_cluster(entry,  num_candidates=10, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    all_near_ids = []

    chunk_size=5

    # Iterate over each cluster index and retrieve 10 nearest samples
    Ds=[]
    Is=[]
    for cluster_id, index in cluster_indexes_global.items():
        # Search for the 10 closest embeddings in the current cluster index
        #near_ids = index.search(embed, max(num_candidates, num_ice)//20+1)[1][0].tolist()
        #all_near_ids.extend(near_ids)

        D, I = index.search(embed, 10)
        #all_near_ids.extend(I[0].tolist())

        ids=random.sample(range(0, len(I[0])), chunk_size)

        #ids=random.sample(range(0, len(I[0])), 2)

        #ids=random.sample([index.id_map.at(i) for i in range(index.id_map.size())] , 5)

        #Ds.extend((-scores)[ids].tolist())
        Ds.extend(D[0][ids].tolist())
        Is.extend(I[0][ids].tolist())
        #Is.extend(ids)


    args = [[entry['metadata']['text'], train_text_list_global[id]] for id in Is]
    text_1, text_2 = zip(*args)

    # Tokenize the pairs
    with torch.no_grad():
        pt_batch = nli_tokenizer_global(list(text_1), list(text_2), padding=True, truncation=True, max_length=512, return_tensors="pt")
        pt_batch = {k: v.to('cuda') for k, v in pt_batch.items()}
        pt_outputs = nli_model_global(**pt_batch)
        pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
        scores=pt_predictions.cpu().detach().numpy()[:,1]

    ids=find_top_indices(-scores,chunk_size=chunk_size, top_n=chunk_size)

    Ds=[-scores[i] for i in ids]
    Is=[Is[i] for i in ids]

    # Zip distances and indices together
    #combined = list(zip(Ds, Is))
    #sorted_combined = sorted(combined, key=lambda x: x[0])
    #sorted_D, sorted_I = zip(*sorted_combined)
    #sorted_D = np.array(sorted_D)
    #sorted_I = np.array(sorted_I)



    all_near_ids=Is

    while -1 in all_near_ids:
        all_near_ids.remove(-1)

    #print(Ds)
    #print(all_near_ids)

    np.savez('selected_data.npz', data_ids=all_near_ids)
    #all_near_ids = random.sample(all_near_ids, len(all_near_ids))



    entry = entry['entry']
    entry['ctxs'] = all_near_ids[:num_ice]
    entry['ctxs_candidates'] = [[i] for i in all_near_ids[:num_candidates]]

    return entry

def knn(entry, num_candidates=1, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, max(num_candidates, num_ice)+1)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids

    entry = entry['entry']
    entry['ctxs'] = near_ids[:num_ice]
    entry['ctxs_candidates'] = [[i] for i in near_ids[:num_candidates]]
    return entry


def compute_similarity_with_encoder(entry_embed, all_embeds, encoder):
    """
    Computes the similarity of `entry_embed` with all embeddings in `all_embeds` using the provided encoder.

    :param entry_embed: The embedding of the current entry.
    :param all_embeds: A matrix of embeddings for all entries.
    :param encoder: The encoder function that computes similarity.
    :return: List of indices sorted by similarity.
    """
    similarities = []
    for other_embed in all_embeds:
        # Assuming `encoder` takes two embeddings and returns a similarity score
        similarity = encoder(entry_embed, other_embed)
        similarities.append(similarity)
    return np.argsort(similarities)[::-1]  # Sort indices by decreasing similarity

def knn_nli(entry, all_entries, encoder, num_candidates=1, num_ice=1, is_train=True):
    """
    Modified kNN function using a custom similarity encoder.

    :param entry: Dictionary containing 'embed' key with the embedding and 'entry'.
    :param all_entries: List of all entries including their 'embed' and 'entry'.
    :param encoder: Function to compute similarity between two embeddings.
    :param num_candidates: Number of top candidates to return.
    :param num_ice: Number of ICE candidates to return.
    :param is_train: Boolean indicating if this is a training scenario.
    """
    entry_embed = np.expand_dims(entry['embed'], axis=0)
    all_embeds = np.array([e['embed'] for e in all_entries])

    # Compute similarities and get the sorted indices of the entries
    sorted_indices = compute_similarity_with_encoder(entry_embed, all_embeds, encoder)

    # Handle training scenario where the first element is itself
    if is_train:
        sorted_indices = sorted_indices[1:]

    entry['ctxs'] = sorted_indices[:num_ice].tolist()
    entry['ctxs_candidates'] = [[idx] for idx in sorted_indices[:num_candidates]]
    return entry





def get_kernel(embed, candidates, scale_factor):
    near_reps = np.stack([index_global.index.reconstruct(i) for i in candidates], axis=0)
    # normalize first
    embed = embed / np.linalg.norm(embed)
    near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)

    rel_scores = np.matmul(embed, near_reps.T)[0]
    # to make kernel-matrix non-negative
    rel_scores = (rel_scores + 1) / 2
    # to prevent overflow error
    rel_scores -= rel_scores.max()
    # to balance relevance and diversity
    rel_scores = np.exp(rel_scores / (2 * scale_factor))
    sim_matrix = np.matmul(near_reps, near_reps.T)
    # to make kernel-matrix non-negative
    sim_matrix = (sim_matrix + 1) / 2
    kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
    return near_reps, rel_scores, kernel_matrix


def random_sampling(num_total, num_ice, num_candidates, pre_results=None):
    ctxs_candidates_idx = [] if pre_results is None else pre_results
    while len(ctxs_candidates_idx) < num_candidates:
        # ordered by sim score
        samples_ids = np.random.choice(num_total, num_ice, replace=False).tolist()
        samples_ids = sorted(samples_ids)
        if samples_ids not in ctxs_candidates_idx:
            ctxs_candidates_idx.append(samples_ids)
    return ctxs_candidates_idx


def dpp(entry, num_candidates=1, num_ice=1, mode="map", dpp_topk=100, scale_factor=0.1):
    candidates = knn(entry, num_ice=dpp_topk)['ctxs']
    embed = np.expand_dims(entry['embed'], axis=0)
    near_reps, rel_scores, kernel_matrix = get_kernel(embed, candidates, scale_factor)

    if mode == "cand_random" or np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
        if np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
            logging.info("Inf or NaN detected in Kernal_matrix, using random sampling instead!")
        topk_results = list(range(num_ice))
        ctxs_candidates_idx = [topk_results]
        ctxs_candidates_idx = random_sampling(num_total=dpp_topk,  num_ice=num_ice,
                                              num_candidates=num_candidates,
                                              pre_results=ctxs_candidates_idx)
    elif mode == "pure_random":
        ctxs_candidates_idx = [candidates[:num_ice]]
        ctxs_candidates_idx = random_sampling(num_total=index_global.ntotal,  num_ice=num_ice,
                                              num_candidates=num_candidates,
                                              pre_results=ctxs_candidates_idx)
        entry = entry['entry']
        entry['ctxs'] = ctxs_candidates_idx[0]
        entry['ctxs_candidates'] = ctxs_candidates_idx
        return entry
    elif mode == "cand_k_dpp":
        topk_results = list(range(num_ice))
        ctxs_candidates_idx = [topk_results]
        ctxs_candidates_idx = k_dpp_sampling(kernel_matrix=kernel_matrix, rel_scores=rel_scores,
                                             num_ice=num_ice, num_candidates=num_candidates,
                                             pre_results=ctxs_candidates_idx)
    else:
        # MAP inference
        map_results = fast_map_dpp(kernel_matrix, num_ice)
        map_results = sorted(map_results)
        ctxs_candidates_idx = [map_results]

    ctxs_candidates = []
    for ctxs_idx in ctxs_candidates_idx:
        ctxs_candidates.append([candidates[i] for i in ctxs_idx])
    assert len(ctxs_candidates) == num_candidates

    entry = entry['entry']
    entry['ctxs'] = ctxs_candidates[0]
    entry['ctxs_candidates'] = ctxs_candidates
    return entry


@hydra.main(config_path="configs", config_name="dense_retriever")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    dense_retriever = DenseRetriever(cfg)
    dense_retriever.find()


if __name__ == "__main__":
    main()
