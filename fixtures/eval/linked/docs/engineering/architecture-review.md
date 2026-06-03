# Architecture Review

The architecture review describes a latency spike around reads after a schema change.
The likely follow-up is [[cache-invalidation]], because stale entries and refresh timing hide inside the data path.
