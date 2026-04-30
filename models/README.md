# Model Artifacts

Training produces `.pth` checkpoint files. The local development folder contained multiple checkpoints of roughly 272 MB each, so they are intentionally excluded from git.

Recommended sharing options:

1. Keep checkpoints in the project Google Drive folder for controlled access.
2. Publish selected checkpoints through GitHub Releases if sharing is approved.
3. Include checksum, training commit, and class mapping with every released checkpoint.

Expected class order for the public four-class model:

```text
ADM, PanIN_LG, PanIN_HG, Other
```
