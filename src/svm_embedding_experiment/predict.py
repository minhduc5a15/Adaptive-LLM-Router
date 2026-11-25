"""Nạp artifact và định tuyến prompt mới; không gọi LLM."""
import argparse
import json

from src.embedding_experiment.encoder import FrozenEncoder
from src.embedding_experiment.inference import positive_probability, route_scores
from src.router import MODEL_NAMES
from src.svm_embedding_experiment.common import DEFAULT_RUN, VARIANTS, MODES
from src.svm_embedding_experiment.evaluate import load_locked
from src.svm_embedding_experiment.features import input_frame


def predict(bundle, prompts, variant, mode='tuned_router', encoder=None):
    prompts = list(prompts)
    vectors = None
    if variant != 'tfidf':
        encoder = encoder or FrozenEncoder()
        if encoder.manifest != bundle['encoder']:
            raise ValueError('Encoder khác lúc training')
        vectors, _ = encoder.encode(prompts, progress=False)
    frame = input_frame(prompts, vectors)
    svm = positive_probability(bundle['svms'][variant], frame)
    tree = positive_probability(bundle['tree'], prompts)
    probability, routes = route_scores(tree, svm, bundle['policies'][variant][mode])
    return probability, routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default=DEFAULT_RUN)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--mode', choices=list(MODES), default='tuned_router')
    parser.add_argument('--prompt', required=True)
    args = parser.parse_args()
    _, _, _, bundle = load_locked(args.run_name)
    variant = args.variant or bundle['selected_variant']
    probabilities, routes = predict(bundle, [args.prompt], variant, args.mode)
    print(json.dumps({'variant': variant, 'mode': args.mode, 'probability': float(probabilities[0]),
                      'model': MODEL_NAMES[int(routes[0])], 'note': 'Chỉ định tuyến; chưa gọi LLM.'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
