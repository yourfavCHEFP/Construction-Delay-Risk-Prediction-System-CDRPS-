# Contributing to CDRPS

Thank you for your interest in contributing to the Construction Delay Risk Prediction System.

## Workflow

1. Create a branch from `main`.
2. Make your changes in a focused commit.
3. Run the dashboard or relevant checks locally.
4. Push the branch to GitHub.
5. Open a pull request with a clear summary and validation notes.

## Branch Naming

- `feature/<short-description>` for new functionality
- `fix/<short-description>` for bug fixes
- `docs/<short-description>` for documentation updates
- `perf/<short-description>` for performance improvements

## Code Standards

- Keep changes small and easy to review.
- Prefer readable, modular Python.
- Avoid hard-coded paths or values when a parameter will do.
- Use caching for expensive dashboard operations.
- Add validation or error handling around file input and data processing.

## Commit Message Format

Use a short type prefix followed by a concise summary:

```text
<type>: <short summary>
```

Common types:

- `feat` - new feature
- `fix` - bug fix
- `docs` - documentation update
- `refactor` - code restructuring without behavior change
- `perf` - performance improvement
- `test` - test changes
- `chore` - maintenance work

Example:

```text
feat: add Streamlit dashboard entry point
```

## Pull Request Checklist

- Describe what changed.
- Explain why it changed.
- Include screenshots for UI work when relevant.
- Note how it was tested.
- Link any related issue or task.