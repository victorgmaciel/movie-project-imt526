# Movie Project — IMT 526 Group 1

Group 1 Reddit + TMDB movie audience signal project.
This project tests whether pre-release Reddit discussion semantics correlate with TMDB movie outcomes.

## Overview

We use Reddit post/comment data, YouTube trailer comments, and TMDB metadata to build a pipeline that predicts box office revenue from pre-release audience signals. LLM inference (Qwen2.5-7B-Instruct) is run on the UW Tillicum HPC cluster (H200 GPUs) to extract semantic features from text data.

## Data Sources

- **Reddit** — pre-release discussion threads scraped by movie title
- **YouTube Data API** — trailer comment sentiment and engagement
- **TMDB API** — movie metadata, revenue, popularity scores

## Project Structure
