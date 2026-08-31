"""B5 -- Relationship / graph features.

Builds a lightweight transaction graph in NetworkX and extracts
structural features. No full GNN -- engineered graph features fed
into the tabular model get 80% of the value for 10% of the effort.
"""

import logging
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger("evo-pay.graph")


def build_transaction_graph(df: pd.DataFrame) -> nx.Graph:
    """Build bipartite graph: customers <-> devices, customers <-> merchants.

    Node types are distinguished by prefix:
      - 'c:' for customers
      - 'd:' for devices
      - 'm:' for merchants

    Edge attributes include transaction count and total amount.
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        cid = f"c:{row['customer_id']}"
        did = f"d:{row.get('device_id', 'unknown')}"
        mid = f"m:{row.get('merchant_id', 'unknown')}"

        # Customer <-> Device edge
        if G.has_edge(cid, did):
            G[cid][did]["tx_count"] += 1
            G[cid][did]["total_amount"] += row.get("amount", 0)
        else:
            G.add_edge(cid, did, tx_count=1, total_amount=row.get("amount", 0))

        # Customer <-> Merchant edge
        if G.has_edge(cid, mid):
            G[cid][mid]["tx_count"] += 1
            G[cid][mid]["total_amount"] += row.get("amount", 0)
        else:
            G.add_edge(cid, mid, tx_count=1, total_amount=row.get("amount", 0))

        # Set node attributes
        G.nodes[cid]["type"] = "customer"
        G.nodes[cid]["is_fraud"] = G.nodes[cid].get("is_fraud", 0) or int(row.get("is_fraud", 0))
        G.nodes[did]["type"] = "device"
        G.nodes[mid]["type"] = "merchant"

    logger.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def compute_graph_features(
    customer_id: str,
    graph: nx.Graph,
    fraud_labels: Optional[dict] = None,
) -> dict:
    """Extract graph-based features for a customer.

    Args:
        customer_id: Raw customer ID (without 'c:' prefix).
        graph: Transaction graph from build_transaction_graph().
        fraud_labels: Optional dict mapping 'c:customer_id' -> is_fraud.

    Returns dict with:
        shared_device_count: How many other customers share a device with this customer
        device_customer_degree: Max degree of any device this customer uses
        merchant_fraud_rate: Fraction of this customer's merchants that are
            also used by known-fraud customers
        customer_degree: Total number of connections for this customer
        is_part_of_dense_cluster: Whether the customer's local clustering
            coefficient is above the 90th percentile
    """
    cid = f"c:{customer_id}"

    if cid not in graph:
        return {
            "shared_device_count": 0,
            "device_customer_degree": 0,
            "merchant_fraud_rate": 0.0,
            "customer_degree": 0,
            "is_part_of_dense_cluster": 0,
        }

    neighbors = list(graph.neighbors(cid))
    devices = [n for n in neighbors if n.startswith("d:")]
    merchants = [n for n in neighbors if n.startswith("m:")]

    # Shared device count: other customers connected to the same devices
    shared_customers = set()
    max_device_degree = 0
    for dev in devices:
        dev_neighbors = [n for n in graph.neighbors(dev) if n.startswith("c:") and n != cid]
        shared_customers.update(dev_neighbors)
        max_device_degree = max(max_device_degree, graph.degree(dev))

    # Merchant fraud rate: fraction of merchants also used by fraud customers
    merchant_fraud_rate = 0.0
    if merchants and fraud_labels:
        fraud_merchant_count = 0
        for m in merchants:
            m_customers = [n for n in graph.neighbors(m) if n.startswith("c:") and n != cid]
            if any(fraud_labels.get(c, 0) for c in m_customers):
                fraud_merchant_count += 1
        merchant_fraud_rate = fraud_merchant_count / len(merchants) if merchants else 0.0

    # Clustering coefficient
    try:
        clustering = nx.clustering(graph, cid)
    except Exception:
        clustering = 0.0

    return {
        "shared_device_count": len(shared_customers),
        "device_customer_degree": max_device_degree,
        "merchant_fraud_rate": round(merchant_fraud_rate, 4),
        "customer_degree": graph.degree(cid),
        "is_part_of_dense_cluster": int(clustering > 0.3),
    }


def compute_graph_features_batch(
    df: pd.DataFrame,
    graph: Optional[nx.Graph] = None,
) -> pd.DataFrame:
    """Compute graph features for all customers in the dataset.

    Args:
        df: Transaction DataFrame.
        graph: Pre-built graph, or None to build from df.

    Returns:
        DataFrame with customer_id and graph feature columns.
    """
    if graph is None:
        graph = build_transaction_graph(df)

    # Build fraud labels from graph node attributes
    fraud_labels = {
        n: graph.nodes[n].get("is_fraud", 0)
        for n in graph.nodes
        if n.startswith("c:")
    }

    # Compute per-customer features
    customers = df["customer_id"].unique()
    rows = []
    for cid in customers:
        feats = compute_graph_features(cid, graph, fraud_labels)
        feats["customer_id"] = cid
        rows.append(feats)

    return pd.DataFrame(rows)
