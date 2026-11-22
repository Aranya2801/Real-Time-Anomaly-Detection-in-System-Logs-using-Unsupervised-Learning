# Per-dataset masking regexes and Drain hyperparameters.
# These follow the settings used in the LogPAI parser benchmark (Zhu et al., ICSE-SEIP 2019)
# so that grouping accuracy is comparable with numbers reported there.

PARSER_SETTINGS = {
    "HDFS": dict(masks=[r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"], sim_threshold=0.5, depth=4),
    "Hadoop": dict(masks=[r"(\d+\.){3}\d+"], sim_threshold=0.5, depth=4),
    "Spark": dict(masks=[r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"], sim_threshold=0.5, depth=4),
    "Zookeeper": dict(masks=[r"(/|)(\d+\.){3}\d+(:\d+)?"], sim_threshold=0.5, depth=4),
    "BGL": dict(masks=[r"core\.\d+"], sim_threshold=0.5, depth=4),
    "HPC": dict(masks=[r"=\d+"], sim_threshold=0.5, depth=4),
    "Thunderbird": dict(masks=[r"(\d+\.){3}\d+"], sim_threshold=0.5, depth=4),
    "Windows": dict(masks=[r"0x.*?\s"], sim_threshold=0.7, depth=5),
    "Linux": dict(masks=[r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"], sim_threshold=0.39, depth=6),
    "Android": dict(
        masks=[r"(/[\w-]+)+", r"([\w-]+\.){2,}[\w-]+",
               r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b"],
        sim_threshold=0.2, depth=6,
    ),
    "HealthApp": dict(masks=[], sim_threshold=0.2, depth=4),
    "Apache": dict(masks=[r"(\d+\.){3}\d+"], sim_threshold=0.5, depth=4),
    "Proxifier": dict(
        masks=[r"<\d+\ssec", r"([\w-]+\.)+[\w-]+(:\d+)?", r"\d{2}:\d{2}(:\d{2})*", r"[KGTM]B"],
        sim_threshold=0.6, depth=3,
    ),
    "OpenSSH": dict(masks=[r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"], sim_threshold=0.6, depth=5),
    "OpenStack": dict(masks=[r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"], sim_threshold=0.5, depth=5),
    "Mac": dict(masks=[r"([\w-]+\.){2,}[\w-]+"], sim_threshold=0.7, depth=6),
}
