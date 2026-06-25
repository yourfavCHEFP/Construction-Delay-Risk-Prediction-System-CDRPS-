# Branching Strategy

Use a simple Git workflow that keeps `main` stable and reviewable.

## Branch Roles

- `main` - production-ready code
- `feature/*` - new features and dashboard work
- `fix/*` - bug fixes
- `docs/*` - documentation changes
- `perf/*` - performance-focused updates

## Recommended Flow

1. Branch from `main`.
2. Implement one logical change per branch.
3. Commit with a clear message.
4. Push the branch to GitHub.
5. Open a pull request into `main`.
6. Merge after review and validation.

## Examples

- `feature/streamlit-dashboard`
- `fix/csv-encoding-detection`
- `docs/readme-cleanup`