# Video Clone Colab Runtime

Repo này chỉ chứa phần runtime cần thiết để chạy backend Video Clone trên Google Colab.

Không cần đưa toàn bộ code desktop, Electron, frontend, build assets hoặc keygen lên Colab.

## File cần có

- `backend/`
- `omnivoice/`
- `pyproject.toml`
- `alembic.ini`
- `Video_Clone_Douyin_Colab.ipynb`
- `LICENSE`

## Cách cập nhật từ repo desktop

Chạy trong repo `Video-Clone-Douyin-main`:

```powershell
.\scripts\export-colab-runtime.ps1 -Destination "C:\path\to\videocolab" -Clean
```

Sau đó commit/push thư mục `videocolab` lên:

```text
https://github.com/nqthaivl/videocolab.git
```
