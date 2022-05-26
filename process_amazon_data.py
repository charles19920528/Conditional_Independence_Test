import pandas as pd
import numpy as np

def load_fashion_data(review_json_path, meta_json_path, reviewer_k, meta_k):
    review_dt = pd.read_json(review_json_path, lines=True)
    meta_dt = pd.read_json(meta_json_path, lines=True)

    more_than_k_reviewer_boolean = review_dt["reviewerID"].value_counts() >= reviewer_k
    more_than_k_reviewer_index = more_than_k_reviewer_boolean.index[more_than_k_reviewer_boolean]

    more_than_k_asin_boolean = review_dt["asin"].value_counts() >= meta_k
    more_than_k_asin_index = more_than_k_asin_boolean.index[more_than_k_asin_boolean]

    keep_boolean = review_dt["reviewerID"].apply(lambda x: x in more_than_k_reviewer_index) & \
                   review_dt["asin"].apply(lambda x: x in more_than_k_asin_index)

    review_dt = review_dt[keep_boolean]
    review_dt = review_dt.drop(["reviewTime", "reviewerName", "unixReviewTime", "image", "style"], axis=1)

    # process metadata
    meta_dt = meta_dt[meta_dt["rank"].notna()]
    meta_dt["rank"] = meta_dt["rank"].apply(lambda x: x[0] if type(x) == list else x)
    meta_dt["rank"] = meta_dt["rank"].str.split("in").str[0].str.replace(",", "")
    meta_dt["rank"] = meta_dt["rank"].str.split("#").apply(lambda x: x[0] if len(x) == 1 else x[1])
    meta_dt["rank"] = meta_dt["rank"].astype(float)

    meta_dt = meta_dt[['title', 'brand', 'rank', 'asin', 'imageURL', 'description', 'price']]
    final_data = review_dt.merge(meta_dt, on="asin", how="left")

    return final_data