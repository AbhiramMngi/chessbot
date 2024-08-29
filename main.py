from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Task Input")
    parser.add_argument("--task", type=str,help="Task to be performed")
    args = parser.parse_args()

    if args.task == "train":
        pass
    elif args.task == "test":
        pass
    elif args.task == "self_play":
        pass
    elif args.task == "open_game":
        pass 
    else:
        parser.error("Invalid task. Please choose from: train, test, self_play, or open_game.")
        