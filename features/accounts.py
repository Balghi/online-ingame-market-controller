import pandas as pd
import numpy as np

def make_account_daily_features(trades: pd.DataFrame) -> pd.DataFrame:
    T = trades.copy()
    T["ts"] = pd.to_datetime(T["ts"])
    T["day"] = T["ts"].dt.floor("D")

    # price baseline per item for z-scores
    item_med = T.groupby("item_id")["price"].median().rename("item_median")
    TT = T.merge(item_med, on="item_id", how="left")
    TT["price_z"] = (TT["price"] - TT["item_median"]) / (TT["item_median"].replace(0, np.nan))
    TT["price_z"] = TT["price_z"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # buyer & seller views
    buyers = TT.groupby(["buyer_id","day"]).agg(
        b_trades=("trade_id","count"),
        b_items=("item_id","nunique"),
        b_counterparties=("seller_id","nunique"),
        b_price_z_mean=("price_z","mean"),
        b_price_z_max=("price_z","max"),
    ).reset_index().rename(columns={"buyer_id":"player_id"})

    sellers = TT.groupby(["seller_id","day"]).agg(
        s_trades=("trade_id","count"),
        s_items=("item_id","nunique"),
        s_counterparties=("buyer_id","nunique"),
        s_price_z_mean=("price_z","mean"),
        s_price_z_max=("price_z","max"),
    ).reset_index().rename(columns={"seller_id":"player_id"})

    df = pd.merge(buyers, sellers, on=["player_id","day"], how="outer").fillna(0)

    # concentration proxy: how many trades involve top 1 counterparty?
    # build pair counts per day
    pair = TT.groupby(["seller_id","buyer_id","day"])["trade_id"].count().reset_index(name="pair_trades")
    # for each player/day, get max pair_trades
    max_pair_as_seller = pair.groupby(["seller_id","day"])["pair_trades"].max().reset_index().rename(
        columns={"seller_id":"player_id","pair_trades":"max_pair_as_seller"})
    max_pair_as_buyer  = pair.groupby(["buyer_id","day"])["pair_trades"].max().reset_index().rename(
        columns={"buyer_id":"player_id","pair_trades":"max_pair_as_buyer"})
    df = df.merge(max_pair_as_seller, on=["player_id","day"], how="left").merge(
             max_pair_as_buyer, on=["player_id","day"], how="left").fillna(0)

    # simple entropy of hours traded (botty/cluster behavior has low entropy)
    TT["hour"] = TT["ts"].dt.hour
    hour_counts = TT.groupby(["buyer_id","day","hour"])["trade_id"].count().reset_index(name="cnt")
    def entropy(g):
        p = g["cnt"] / g["cnt"].sum()
        return float(-(p*np.log2(p)).sum())
    ent_buyer = (
        hour_counts.groupby(["buyer_id","day"])
        .apply(entropy, include_groups=False)
        .reset_index(name="buyer_hour_entropy")
        .rename(columns={"buyer_id":"player_id"})
    )
    hour_counts_s = TT.groupby(["seller_id","day","hour"])["trade_id"].count().reset_index(name="cnt")
    ent_seller = (
        hour_counts_s.groupby(["seller_id","day"])
        .apply(entropy, include_groups=False)
        .reset_index(name="seller_hour_entropy")
        .rename(columns={"seller_id":"player_id"})
    )
    df = df.merge(ent_buyer, on=["player_id","day"], how="left").merge(ent_seller, on=["player_id","day"], how="left").fillna(0)

    return df
