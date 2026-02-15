try:
    from runai_sharded_hotfix import apply_hotfixes

    apply_hotfixes()
except Exception:
    # Keep Python startup resilient in non-vLLM processes.
    pass
