import os
import re
import requests
import tweepy


def get_x_clients() -> tuple[tweepy.Client, tweepy.API]:
    api_key = os.environ["X_API_KEY"]
    api_secret = os.environ["X_API_SECRET"]
    access_token = os.environ["X_ACCESS_TOKEN"]
    access_secret = os.environ["X_ACCESS_TOKEN_SECRET"]

    client_v2 = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_secret,
    )
    auth_v1 = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
    api_v1 = tweepy.API(auth_v1)
    return client_v2, api_v1


def get_issue(repo: str, issue_number: str, token: str) -> dict:
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def comment_and_close(repo: str, issue_number: str, token: str, message: str) -> None:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    requests.post(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments",
        headers=headers,
        json={"body": message},
        timeout=30,
    ).raise_for_status()
    requests.patch(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}",
        headers=headers,
        json={"state": "closed"},
        timeout=30,
    ).raise_for_status()


def post_comment_only(repo: str, issue_number: str, token: str, message: str) -> None:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    requests.post(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments",
        headers=headers,
        json={"body": message},
        timeout=30,
    ).raise_for_status()


def parse_issue(body: str) -> tuple[list[str], dict[str, str]]:
    chart_paths = re.findall(r"\!\[.*?\]\(https://raw\.githubusercontent\.com/[^/]+/[^/]+/[^/]+/(charts/[^)]+\.png)\)", body)
    options: dict[str, str] = {}
    for num in ("1", "2", "3"):
        match = re.search(rf"\*\*{num}:\*\*\s*(.+)", body)
        if match:
            options[num] = match.group(1).strip()
    return chart_paths, options


def main() -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    issue_number = os.environ["ISSUE_NUMBER"]
    comment_body = os.environ.get("COMMENT_BODY", "").strip()

    if not comment_body:
        return

    if comment_body.lower() in {"skip", "ignore", "close"}:
        comment_and_close(repo, issue_number, token, "Skipped. Closing NFL draft issue without posting.")
        return

    issue = get_issue(repo, issue_number, token)
    chart_paths, options = parse_issue(issue.get("body", ""))
    post_text = options.get(comment_body, comment_body)

    if len(post_text) > 280:
        post_comment_only(
            repo,
            issue_number,
            token,
            f"Caption exceeds 280 characters ({len(post_text)} chars). Comment a shorter version or pick `1`, `2`, or `3`.",
        )
        return

    client_v2, api_v1 = get_x_clients()
    media_ids = []
    for path in chart_paths:
        if os.path.exists(path):
            media = api_v1.media_upload(filename=path)
            media_ids.append(media.media_id_string)

    try:
        response = client_v2.create_tweet(
            text=post_text,
            media_ids=media_ids if media_ids else None,
            user_auth=True,
        )
    except tweepy.TweepyException as e:
        post_comment_only(repo, issue_number, token, f"X API error while publishing (issue kept open):\n\n> {e}")
        raise

    posted_url = f"https://x.com/pixelgaragedata/status/{response.data['id']}"
    comment_and_close(
        repo,
        issue_number,
        token,
        f"Published NFL Power Rankings to X:\n{posted_url}\n\n> {post_text}",
    )


if __name__ == "__main__":
    main()
