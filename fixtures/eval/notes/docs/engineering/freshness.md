# Index Freshness

Search should notice when files changed and refresh the local index before it
answers. A watcher can keep the index warm, but self-healing search protects the
default workflow when no background process is running.

