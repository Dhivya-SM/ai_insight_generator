# scripts/create_jira_ticket.py

import requests
from requests.auth import HTTPBasicAuth

JIRA_EMAIL = "dhivyaece2015@gmail.com"
JIRA_API_TOKEN = "ATATT3xFfGF0WUTvek0FrxkdUYYR0l-Q96T75o0tvnoBxKVc0bJFUhqFMo_m830qzojOXbunLxNSf4_XG82ic4uYEDO480oTC5FMNiFWAfGb3nk8Q_qIYBrVKUnSaT-795tGOHAxMwsFjOhJLxzRyFMyq6Pw4i5r2ptCfw3aXX6fdaqQs-ZxtVs=DBC35E0F"
JIRA_URL = "https://dhivya-sm-testing.atlassian.net"
JIRA_PROJECT_KEY = "SCRUM"  # Use your actual project key

def create_jira_ticket(summary, description):
    url = f"{JIRA_URL}/rest/api/3/issue"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "fields": {
            "project": {
                "key": JIRA_PROJECT_KEY
            },
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": description
                            }
                        ]
                    }
                ]
            },
            "issuetype": {
                "name": "Task"
            }
        }
    }

    response = requests.post(url, json=payload, headers=headers, auth=auth)

    if response.status_code == 201:
        issue_key = response.json().get("key")
        print(f"✅ Ticket created: {issue_key}")
        return issue_key
    else:
        print(f"❌ Failed to create ticket: {response.status_code} {response.text}")
        return None
