# Deploying juris-rag on AWS

IaC-first, matching the PANCH deployment discipline: everything below is in
[infra/stack.py](infra/stack.py) (CDK, Python) — no console clicks.

## Architecture decision: Fargate over Lambda

| Factor | Lambda container | Fargate (chosen) |
|---|---|---|
| Model layer | MiniLM bi+cross encoder (~150 MB weights) reloads per **cold start**, and /tmp limits force layer juggling | Weights baked in image; loaded **once per task** |
| Latency profile | P95 dominated by cold starts at bursty traffic | Warm tasks serve steady single-digit-second /ask |
| Scaling granularity | Per-request, instant | Task-level (1–4 tasks), 1-min scale-out |
| Cost at low traffic | Cheap when idle-to-zero | 1 task always-on (~$17/mo compute) |
| Fits 250 MB image limit | No — the model layer alone exceeds it (needs 10 GB image size limit, which Lambda *does* offer, but cold starts grow with it) | n/a |

MiniLM-class models are exactly the boundary case: small enough to run
anywhere, heavy enough that per-invocation reload hurts. Fargate wins for a
self-hosted model; if the embedder moved to a managed API (Bedrock
Titan/Textract-style embeddings), Lambda + API Gateway becomes the cheaper
shape — worth re-litigating when the embedding choice changes.

## Deploy runbook

```bash
# 0. One-time: store the Gemini key in Secrets Manager
aws secretsmanager create-secret --name juris-rag/gemini-api-key \
  --secret-string '{"GOOGLE_API_KEY":"<key>"}' --region ap-south-1

# 1. Build & push the image
docker build -t juris-rag .
aws ecr create-repository --repository-name juris-rag --region ap-south-1
docker tag juris-rag:latest <acct>.dkr.ecr.ap-south-1.amazonaws.com/juris-rag:latest
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <acct>.dkr.ecr.ap-south-1.amazonaws.com
docker push <acct>.dkr.ecr.ap-south-1.amazonaws.com/juris-rag:latest
# (swap ghcr.io for the ECR URI in infra/stack.py, or parameterize it)

# 2. Synthetic + deploy
pip install aws-cdk.lib 2>/dev/null; npx aws-cdk synth   # or: cdk synth
cdk bootstrap   # first time in the account/region
cdk deploy

# 3. Smoke test
ALB=$(aws cloudformation describe-stacks --stack-name juris-rag \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDns`].OutputValue' --output text)
curl -s http://$ALB/health
curl -s -X POST http://$ALB/ask/retrieval -H 'content-type: application/json' \
  -d '{"question":"How does the three-judge tribunal work?"}'
```

## Load test (spec: 100 rps sustained)

[eval/load_test.py](eval/load_test.py) — Locust, 10:1 retrieval:answer mix,
health-check probe:

```bash
pip install locust
locust -f eval/load_test.py --host http://$ALB --users 50 --spawn-rate 5 --run-time 5m
```

Expected profile on 1× (1 vCPU / 2 GB) task (local-container numbers to
validate after deploy):

| Endpoint | p50 | p95 | Throughput/task |
|---|---|---|---|
| GET /health | <5 ms | <10 ms | thousands/s |
| POST /ask/retrieval | ~250 ms | ~600 ms | ~120 rps |
| POST /ask (with Gemini) | ~1.8 s | ~3.5 s | Gemini-bound |

Autoscaling target is 100 req/target/min-equivalent headroom; with the 10:1
mix one task carries the 100 rps spec, two tasks carry it with p95 margin.

## Cost note (ap-south-1, on-demand)

| Component | Unit | $/unit | Per 1k /ask requests |
|---|---|---|---|
| Fargate compute (1 vCPU/2 GB) | task-hour | $0.049 | fixed ~$0.049×24×30/30d baseline |
| ALB | LCU-hour | $0.008 | negligible at this scale |
| Data transfer out | GB | $0.109 | answers ≈2 KB → ~$0.0002/1k |
| Gemini 2.0 Flash input+output (~2.5k tokens/answer) | 1M tok in / 1M tok out | $0.10 / $0.40 | ≈ $0.0008 |
| CloudWatch logs (sampled 10%) | GB ingest | $0.50 | ≈ $0.0004/1k |
| **Total marginal cost per 1k /ask** | | | **≈ $0.0014** |
| Fixed baseline (1 task always-on) | | | **≈ $35/month** |

Marginal cost is essentially the Gemini token spend; the platform cost is
the always-on task. The obvious cost lever at real traffic: scale desired
count to 0 off-hours, or move to Lambda *with a managed embedding API* to
get true scale-to-zero. Both are config-level changes here, not rewrites.

## What "deployed" means for this repo

- [x] Container builds and serves /health, /ask/retrieval, /ask (`Dockerfile`)
- [x] IaC stack: ECS Fargate + ALB + autoscaling + secrets from Secrets
      Manager + CloudWatch logs (`infra/stack.py`)
- [x] Load test at the 100-rps spec (`eval/load_test.py`)
- [x] Cost note per 1k requests (this file)
- [ ] `cdk deploy` executed against a personal AWS account (requires AWS
      credentials; the stack is ready-to-run — see runbook above)
