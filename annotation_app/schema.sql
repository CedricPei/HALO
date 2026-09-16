CREATE TABLE IF NOT EXISTS annotation_users (
    user_id TEXT PRIMARY KEY,
    access_code TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS annotation_assignments (
    user_id TEXT NOT NULL,
    review_type TEXT NOT NULL CHECK (review_type IN ('audit', 'eval')),
    item_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'assigned'
        CHECK (status IN ('assigned', 'submitted')),
    assigned_at TEXT NOT NULL,
    submitted_at TEXT,
    PRIMARY KEY (user_id, review_type, item_id),
    FOREIGN KEY (user_id) REFERENCES annotation_users(user_id)
);

CREATE INDEX IF NOT EXISTS annotation_assignments_user_type_position
    ON annotation_assignments(user_id, review_type, position);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    review_type TEXT NOT NULL CHECK (review_type IN ('audit', 'eval')),
    item_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, review_type, item_id),
    FOREIGN KEY (user_id) REFERENCES annotation_users(user_id)
);

CREATE INDEX IF NOT EXISTS annotations_lookup
    ON annotations(user_id, review_type, item_id);

CREATE TABLE IF NOT EXISTS annotation_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotation_assignment_repairs (
    repair_id TEXT PRIMARY KEY,
    review_type TEXT NOT NULL CHECK (review_type IN ('audit', 'eval')),
    created_at TEXT NOT NULL,
    selected_users_json TEXT NOT NULL,
    previous_assignments_json TEXT NOT NULL,
    saved_annotation_count INTEGER NOT NULL
);

ALTER TABLE annotation_users DROP COLUMN IF EXISTS display_name;
ALTER TABLE annotation_users DROP COLUMN IF EXISTS role;
ALTER TABLE annotation_users DROP COLUMN IF EXISTS active;
ALTER TABLE annotation_users ADD COLUMN IF NOT EXISTS access_code TEXT NOT NULL DEFAULT '';
ALTER TABLE annotation_users ALTER COLUMN access_code DROP DEFAULT;
ALTER TABLE annotation_users DROP COLUMN IF EXISTS access_code_hash;

ALTER TABLE annotation_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotation_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotation_migrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotation_assignment_repairs ENABLE ROW LEVEL SECURITY;
