import benchmark as bench
from benchmark.utils.config_utils import (
    load_config,
    merge_args_to_config,
    override_config,
    print_config,
)


def main(args):
    from benchmark.models.musichubert_hf.extract_bert_features import (
        main as extract_hubert_features_main,  # mert
    )

    config = load_config(args.config, namespace=True)

    if args.override is not None and args.override.lower() != "none":
        override_config(args.override, config)

    config = merge_args_to_config(args, config)

    print_config(config)

    representation_name = config.dataset.pre_extract.feature_extractor.pretrain.name
    extract_main = bench.NAME_TO_EXTRACT_FEATURES_MAIN[representation_name]
    extract_main = eval(extract_main)
    extract_main(config)
