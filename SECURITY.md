# Security

## Credentials

The runners read `DIGITALOCEAN_INFERENCE_API_KEY` from the process environment. They must never write it to plans, request journals, reports, issue text, or commits.

If a real key is ever committed or pasted into a public location, revoke it immediately in DigitalOcean before attempting repository cleanup. Git history should be treated as public and persistent.

## Reporting a vulnerability

Open a private GitHub security advisory for vulnerabilities in the harness. Do not include live credentials or sensitive response data in an issue.
