# Movie Project — IMT 526 Group 1

Group 1 Reddit + TMDB movie audience signal project. This project tests whether pre-release Reddit discussion semantics correlate with TMDB movie outcomes.

## Overview

We use Reddit post/comment data, YouTube trailer comments, and TMDB metadata to build a pipeline that predicts box office revenue from pre-release audience signals. LLM inference (Qwen2.5-7B-Instruct) is run on the UW Tillicum HPC cluster (H200 GPUs) to extract semantic features from text data.

## Data Sources

- **Reddit** — pre-release discussion threads scraped by movie title
- **YouTube Data API** — trailer comment sentiment and engagement
- **TMDB API** — movie metadata, revenue, popularity scores

## Project Structure

project_root/
├── data/           # Raw and processed data (not tracked in git)
├── logs/           # Job logs from Slurm/HPC runs
├── outputs/        # Model outputs and predictions
├── scripts/        # Data collection, preprocessing, and inference scripts
├── slurm/          # Slurm job submission scripts for Tillicum
└── Untitled.ipynb  # Exploratory notebook

## Setup

API keys needed (set as environment variables):
TMDB_API_KEY=...
YOUTUBE_API_KEY=...

## Running on Tillicum (UW HPC)

Submit jobs via Slurm:
sbatch slurm/your_job_script.sh

Model used: Qwen2.5-7B-Instruct on H200 GPUs.

## Course Context

University of Washington — IMT 526A (Winter/Spring 2026)
Group 1: Reddit Movie Signals
