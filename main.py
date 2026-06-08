def main():
    """Self-supervised LeJEPA training on flotation plant data."""
    from ss26_ml_prozess.ssl.config import LeJEPAConfig
    from ss26_ml_prozess.ssl.train import run

    cfg = LeJEPAConfig()
    run(cfg)


if __name__ == "__main__":
    main()
