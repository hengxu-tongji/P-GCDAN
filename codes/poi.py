import os
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle as pickle
from tqdm import tqdm

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
import gensim

import trackintel as ti


def _read_poi_files():
    df = pd.read_csv('pois.csv', encoding='ANSI')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LON, df.LAT),crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:32650")
    gdf.to_file(os.path.join("data", "poi", "final_pois.shp"))




def get_poi_representation(method="lda", categories=16):
    # checked: buffer method; transform to final_poi; vector values are different

    # read location and change the geometry columns
    locs = ti.read_locations_csv(os.path.join("data", f"locations_geolife.csv"), index_col="id", crs="EPSG:4326")
    locs = gpd.GeoDataFrame(locs.drop(columns="center"), crs="EPSG:4326", geometry="extent").to_crs("EPSG:32650")

    # drop duplicate index
    locs.drop(columns="user_id", inplace=True)
    locs.reset_index(inplace=True)
    locs.rename(columns={"id": "loc_id"}, inplace=True)

    # read poi file
    poi = gpd.read_file(os.path.join("data", "poi", "final_pois.shp"))
    spatial_index = poi.sindex

    poi_dict_ls = []
    buffer_ls = np.arange(11) * 50
    # buffer_ls = [250]

    for distance in buffer_ls:
        curr_locs = locs.copy()
        ## create buffer for each location
        if distance != 0:
            curr_locs["extent"] = curr_locs["extent"].buffer(distance=distance)

        # get the inside poi within each location
        tqdm.pandas(desc="Generating poi within")
        curr_locs["poi_within"] = curr_locs["extent"].progress_apply(
            _get_inside_pois, poi=poi, spatial_index=spatial_index
        )

        # cleaning and expanding to location_id-poi_id pair
        curr_locs.drop(columns="extent", inplace=True)

        # explode preserves nan - preserves locs with no poi
        locs_poi = curr_locs.explode(column="poi_within")

        # get the poi info from original poi df
        locs_poi = locs_poi.merge(poi[["fid", "CATEGORY", "TYPECODE"]], left_on="poi_within", right_on="fid", how="left")
        locs_poi.drop(columns=["fid"], inplace=True)

        # final cleaning
        valid_pairs = locs_poi.dropna(subset=["poi_within"]).copy()
        valid_pairs["TYPECODE"] = valid_pairs["TYPECODE"].astype(int).astype(str)

        # get the poi representation
        if method == "lda":
            poi_dict = _lda(valid_pairs, categories=categories)
        elif method == "tf_idf":
            poi_dict = _tf_idf(valid_pairs, categories=categories)
        else:
            raise AttributeError

        poi_dict_ls.append(poi_dict)

    ## create the matrix
    idx_max = np.max([poi_dict["index"].max() for poi_dict in poi_dict_ls])
    all_idx = np.arange(idx_max + 1)

    # num_loc*lda_vector*buffer_num
    all_poi = np.zeros([len(all_idx), categories, len(buffer_ls)])

    for i, poi_dict in enumerate(poi_dict_ls):
        for j, idx in enumerate(poi_dict["index"]):
            # print(i, idx)
            all_poi[idx, :, i] = poi_dict["poiValues"][j, :]

    all_poi_dict = {"index": all_idx, "poiValues": all_poi}

    ## save to disk
    with open(os.path.join("data", f"poiValues_{method}_{categories}.pk"), "wb") as handle:
        pickle.dump(all_poi_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _get_inside_pois(df, poi, spatial_index):
    """
    Given one extent (df), return the poi within this extent.
    spatial_index is obtained from poi.sindex to speed up the process.
    """
    possible_matches_index = list(spatial_index.intersection(df.bounds))
    possible_matches = poi.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.within(df)]["fid"].values

    return precise_matches


def _tf_idf(df, categories=8):
    """Note: deal with the manually assigned "category" field."""
    texts = df.groupby("loc_id")["category"].apply(list).to_list()

    dct = Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]

    tfmodel = TfidfModel(corpus)
    vector = tfmodel[corpus]

    # the tf array
    dense_tfvector = gensim.matutils.corpus2dense(vector, num_terms=categories).T
    # the index arr
    index_arr = df.groupby("loc_id").count().reset_index()["loc_id"].values
    return {"index": index_arr, "poiValues": dense_tfvector}


def _lda(df, categories=16):
    """Note: deal with the osm assigned "code" field."""
    texts = df.groupby("loc_id")["TYPECODE"].apply(list).to_list()

    dct = Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]

    lda = LdaModel(corpus, num_topics=categories)
    vector = lda[corpus]

    # the lda array
    dense_ldavector = gensim.matutils.corpus2dense(vector, num_terms=categories).T
    # the index arr
    index_arr = df.groupby("loc_id", as_index=False).count()["loc_id"].values
    return {"index": index_arr, "poiValues": dense_ldavector}


if __name__ == "__main__":
    _read_poi_files()

    get_poi_representation(method="lda", categories=16)
