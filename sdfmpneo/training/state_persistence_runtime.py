from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np

from sdfmpneo.analytic.state_graph import (
    AnalyticStateSource,
    add_response_state,
    has_multi_source_states,
    response_sources,
)


def serialize_graph(graph):
    return [
        {
            "name": node.name,
            "target_mode": int(node.target_mode),
            "sources": [
                {
                    "parents": list(source.parents),
                    "weight": [float(source.weight.real), float(source.weight.imag)],
                }
                for source in response_sources(graph, node)
            ],
        }
        for node in graph.response_nodes
    ]


def load_graph_from_metadata(lambdas, operating_names, nodes):
    from sdfmpneo.analytic.parametric import ParametricAnalyticEvolutionGraph

    graph = ParametricAnalyticEvolutionGraph(lambdas, operating_names)
    for node in nodes:
        if "sources" in node:
            sources = [
                AnalyticStateSource.make(
                    source["parents"], complex(*source["weight"])
                )
                for source in node["sources"]
            ]
            add_response_state(
                graph, node["name"], node["target_mode"], sources
            )
        else:
            graph.add_product_response(
                node["name"],
                node["target_mode"],
                node["parents"],
                complex(*node["weight"]),
            )
    return graph


def install_state_persistence(model_class) -> None:
    if getattr(model_class, "_analytic_state_persistence_installed", False):
        return
    original_save = model_class.save
    original_load = model_class.load.__func__

    def save(self, path):
        if self.graph is None or not has_multi_source_states(self.graph):
            return original_save(self, path)
        with TemporaryDirectory() as directory:
            temporary = Path(directory) / "legacy.npz"
            original_save(self, temporary)
            with np.load(temporary, allow_pickle=False) as data:
                arrays = {key: np.array(data[key]) for key in data.files}
        metadata = json.loads(str(arrays["metadata"]))
        metadata["format_version"] = 2
        metadata["nodes"] = serialize_graph(self.graph)
        arrays["metadata"] = np.array(json.dumps(metadata))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("model_type") == "geometry_research":
                return original_load(cls, path)
            version = int(metadata.get("format_version", 1))
            if version == 1:
                return original_load(cls, path)
            if version != 2:
                raise ValueError("unsupported model format version")
            arrays = {key: np.array(data[key]) for key in data.files}

        legacy = dict(metadata)
        legacy["format_version"] = 1
        legacy["nodes"] = [
            {
                "name": node["name"],
                "target_mode": node["target_mode"],
                "parents": node["sources"][0]["parents"],
                "weight": node["sources"][0]["weight"],
            }
            for node in metadata["nodes"]
        ]
        arrays["metadata"] = np.array(json.dumps(legacy))
        with TemporaryDirectory() as directory:
            temporary = Path(directory) / "compat.npz"
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            model = original_load(cls, temporary)

        model.graph = load_graph_from_metadata(
            model.core.thermal_model.lambdas,
            metadata["operating_names"],
            metadata["nodes"],
        )
        return model

    model_class.save = save
    model_class.load = load
    model_class._analytic_state_persistence_installed = True
