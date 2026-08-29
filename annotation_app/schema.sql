CREATE TABLE IF NOT EXISTS annotation_users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('annotator', 'reviewer', 'admin')),
    access_code_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
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
    state TEXT NOT NULL DEFAULT 'submitted'
        CHECK (state IN ('draft', 'submitted')),
    created_at TEXT NOT NULL,
    UNIQUE (user_id, review_type, item_id),
    FOREIGN KEY (user_id) REFERENCES annotation_users(user_id)
);

CREATE INDEX IF NOT EXISTS annotations_lookup
    ON annotations(user_id, review_type, item_id);

CREATE TABLE IF NOT EXISTS annotation_admin_events (
    event_id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

ALTER TABLE annotation_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotation_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotation_admin_events ENABLE ROW LEVEL SECURITY;
