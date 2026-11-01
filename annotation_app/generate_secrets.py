"""Generate strong values to paste into Streamlit Community Cloud Secrets."""

import secrets


def main() -> None:
    print(f'ADMIN_PASSWORD = "{secrets.token_urlsafe(24)}"')


if __name__ == "__main__":
    main()
