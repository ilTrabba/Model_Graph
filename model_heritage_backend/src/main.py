from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Crea directory uploads se non esiste
os.makedirs('uploads', exist_ok=True)

# Database setup
def init_db():
    conn = sqlite3.connect('database/models.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            parent_id INTEGER,
            FOREIGN KEY (parent_id) REFERENCES models (id)
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/api/models', methods=['GET'])
def get_models():
    conn = sqlite3.connect('database/models.db')
    cursor = conn.execute('SELECT * FROM models')
    models = [{'id': row[0], 'name': row[1], 'filename': row[2], 'upload_date': row[3], 'parent_id': row[4]} for row in cursor.fetchall()]
    conn.close()
    return jsonify(models)

@app.route('/api/models', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    model_name = request.form.get('name', file.filename)
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file.save(f'uploads/{filename}')
    
    # Save to database
    conn = sqlite3.connect('database/models.db')
    conn.execute('INSERT INTO models (name, filename, upload_date) VALUES (?, ?, ?)',
                 (model_name, filename, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Model uploaded successfully'})

if __name__ == '__main__':
    os.makedirs('database', exist_ok=True)
    init_db()
    app.run(host='0.0.0.0', port=5001, debug=True)