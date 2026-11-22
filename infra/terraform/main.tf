# IaC skeleton for the platform's cloud footprint (AWS). Authored and `terraform
# validate`-clean; not applied in CI. Backend/credentials are supplied at apply time.
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "region" {
  type    = string
  default = "eu-west-2"
}

variable "project" {
  type    = string
  default = "mlops-platform"
}

provider "aws" { region = var.region }

# Artifact + model store
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project}-artifacts"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Container registry for the serving image
resource "aws_ecr_repository" "serving" {
  name                 = "${var.project}-serving"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

output "artifacts_bucket" { value = aws_s3_bucket.artifacts.bucket }
output "ecr_repo_url" { value = aws_ecr_repository.serving.repository_url }
