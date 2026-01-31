"""add hnsw index

Revision ID: 9486e5077e65
Revises: 6adc7a7b9526
Create Date: 2026-01-19 17:53:01.896344

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9486e5077e65"
down_revision: str | Sequence[str] | None = "6adc7a7b9526"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_notes_embedding_hnsw
        ON notes
        USING hnsw (embedding vector_cosine_ops);
    """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_notes_embedding_hnsw;")
