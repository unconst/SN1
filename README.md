# SN1

## Validating
Set env vars, chutes api key.
```bash
# Copy .env and fill out validator items
cp .env.example .env
```

Run the validator with docker and watchtower autoupdate.
```bash
# Run the validator with watchtower.
docker-compose down && docker-compose pull && docker-compose up -d && docker-compose logs -f
```