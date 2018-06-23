import argparse


def main(setting_file):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting_file', default='settings.txt', type=str)
    args = parser.parse_args()
    main(args.setting_file)
