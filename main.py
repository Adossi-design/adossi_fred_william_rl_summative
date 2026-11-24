"""Open the 3D farm in the browser, so a farmer can watch or play a maize season."""
import argparse
import urllib.parse
import webbrowser
from pathlib import Path

from farmer_policy import TRAINED_MODELS, best_model
from environment.api import PORT, TESTS, start_server

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["trained", "expert"],
        default="trained",
        help="trained = the best saved agent; expert = the Expert rules farmer (same seed)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", choices=list(TESTS))
    parser.add_argument("--algo")
    parser.add_argument("--exp", type=int)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=1,
        help="training seed of the model (with --algo/--exp)",
    )
    parser.add_argument("--port", type=int, default=PORT, help="port of the local server")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # Pick the farmer that the 3D page shows when it opens.
    if args.policy == "expert":
        model = "expert"
    elif args.algo and args.exp:
        model = f"{args.algo}-{args.exp}-{args.model_seed}"
    else:
        algo, exp = best_model()
        model = algo
        if TRAINED_MODELS.get(algo) != (algo, exp):
            model = f"{algo}-{exp}-1"

    query = {"mode": "watch", "model": model, "seed": args.seed}
    if args.test:
        query["test"] = args.test

    try:
        server, url = start_server(ROOT, args.port)
    except OSError:
        print(f"Port {args.port} is already in use, close the other copy of this demo or choose another port with --port.")
        return
    url = url + "?" + urllib.parse.urlencode(query)
    print("3D view:", url)
    print("Press Ctrl+C to stop the server.")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
