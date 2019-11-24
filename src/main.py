import os
from lib.utils.config import Config
from lib.data.format_data import FormatData

# Initialise variables
rdms = {
    "pearson": {},
    "kernel": {}
}


def main(config):
    FormatData(config, "GOD").format()


if __name__ == "__main__":
    config = Config().parse()
    main(config)
