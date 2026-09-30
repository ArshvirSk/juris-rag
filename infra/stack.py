#!/usr/bin/env python3
"""juris-rag AWS CDK stack — Fargate + ALB deployment.

Deploys the container in Dockerfile as an ECS Fargate service behind an
Application Load Balancer with target-tracking autoscaling and CloudWatch
logs. Chosen over Lambda because the model layer (MiniLM bi-encoder +
cross-encoder, ~150 MB weights) is baked into the image and reloads on
every cold start — Fargate pays that once per scale-out, Lambda per burst.

Synth:  npx cdk synth -c secret_arn=<secretsmanager-arn-for-gemini-key>
Deploy: npx cdk deploy -c secret_arn=...

Secret handling matches PANCH's discipline: the Gemini key is read from
Secrets Manager at container start, never baked into the image or IaC.
"""

from __future__ import annotations

import os

from aws_cdk import (
    App,
    CfnOutput,
    Duration,
    Environment,
    SecretValue,
    Stack,
    StackProps,
)
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

IMAGE_TAG = os.environ.get("JURIS_IMAGE_TAG", "latest")


class JurisRagStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        secret_arn = self.node.try_get_context("secret_arn")
        gemini_secret = (
            secretsmanager.Secret.from_secret_name_v2(
                self, "GeminiKey", "juris-rag/gemini-api-key"
            )
            if not secret_arn
            else secretsmanager.Secret.from_secret_complete_arn(
                self, "GeminiKey", secret_arn
            )
        )

        vpc = ec2.Vpc(self, "JurisVpc", max_azs=2, nat_gateways=1)

        cluster = ecs.Cluster(self, "JurisCluster", vpc=vpc)

        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "JurisService",
            cluster=cluster,
            cpu=1024,                      # 1 vCPU — model load is CPU-bound
            memory_limit_mib=2048,         # MiniLM pairs + ChromaDB fit easily
            desired_count=1,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_registry(
                    f"ghcr.io/arshvirsk/juris-rag:{IMAGE_TAG}"
                ),
                container_name="juris-rag",
                container_port=8080,
                secrets={
                    "GOOGLE_API_KEY": ecs.Secret.from_secrets_manager(
                        gemini_secret
                    ),
                },
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="juris",
                ),
            ),
            public_load_balancer=True,
            health_check_grace_period=Duration.seconds(120),  # model load time
        )

        # Autoscale on request count; scale-in is conservative because of
        # the model-load cost on cold tasks.
        scaling = fargate_service.service.auto_scale_task_count(
            min_capacity=1, max_capacity=4
        )
        scaling.scale_on_request_count(
            "JurisScaling",
            requests_per_target=100,
            target_utilization_percent=70,
            scale_in_cooldown=Duration.minutes(5),
            scale_out_cooldown=Duration.minutes(1),
        )

        CfnOutput(
            self,
            "LoadBalancerDns",
            value=fargate_service.load_balancer.load_balancer_dns_name,
        )
        CfnOutput(self, "ServiceName", value=fargate_service.service.service_name)


app = App()
JurisRagStack(
    app,
    "juris-rag",
    env=Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"],
                    region=os.environ.get("CDK_DEFAULT_REGION", "ap-south-1")),
)
app.synth()
