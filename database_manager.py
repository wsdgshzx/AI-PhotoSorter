import os
import sqlite3
from logger_unit import ConsoleColor, CustomLogger  

class DatabaseManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            db_path = os.path.join(os.getcwd(), 'photo_organizer.db')
            cls._instance = cls(db_path)
        return cls._instance

    def __init__(self, db_path):
        self.logger = CustomLogger('DatabaseManager')
        self.conn = sqlite3.connect(db_path)        
        self._create_tables()
        

    def _create_tables(self):
        try:
            create_table_base = '''
            CREATE TABLE IF NOT EXISTS photo_path (
                md5 TEXT NOT NULL PRIMARY KEY,
                file_path TEXT NOT NULL
            )
            '''
            self.conn.execute(create_table_base)

            # Make (Camera Manufacturer)
            # Model (Camera Model)
            # ExposureTime (Exposure Time)
            # FNumber (Aperture Value)
            # ISO (Sensitivity)
            # DateTimeOriginal (Original Shooting Date and Time)
            # FocalLength (Focal Length)
            # ShutterSpeedValue (Shutter Speed)
            # ApertureValue (Aperture Size)
            # BrightnessValue (Brightness Value)
            # Flash (Flash Status)
            # WhiteBalance (White Balance)
            # MeteringMode (Metering Mode)
            # ExposureMode (Exposure Mode)
            # ExposureProgram (Exposure Program)
            # GPSLatitude (GPS Latitude)
            # GPSLongitude (GPS Longitude)
            # GPSAltitude (GPS Altitude)
            # LensMake (Lens Manufacturer)
            # LensModel (Lens Model)
            create_exif_table = '''CREATE TABLE IF NOT EXISTS exif_data (
                md5 TEXT PRIMARY KEY,
                make TEXT,
                model TEXT,
                exposure_time TEXT,
                f_number TEXT,
                iso INTEGER,
                date_time_original INTEGER,
                focal_length TEXT,
                shutter_speed_value TEXT,
                aperture_value TEXT,
                brightness_value TEXT,
                flash INTEGER,
                white_balance INTEGER,
                metering_mode INTEGER,
                exposure_mode INTEGER,
                exposure_program INTEGER,
                gps_latitude REAL,
                gps_longitude REAL,
                gps_altitude REAL,
                lens_make TEXT,
                lens_model TEXT
            )
            '''
            self.conn.execute(create_exif_table)


            create_table_similarity = '''
            CREATE TABLE IF NOT EXISTS image_similarity (
                md5_1 TEXT,
                md5_2 TEXT,
                similarity REAL,
                PRIMARY KEY (md5_1, md5_2)
            )
            '''
            self.conn.execute(create_table_similarity)

            # Create all score table, integrating scores from various algorithms
            create_table_all_scores = '''
            CREATE TABLE IF NOT EXISTS all_scores (
                md5 TEXT KEY,
                metric_name TEXT,
                score REAL NOT NULL,
                PRIMARY KEY (md5, metric_name)
            )
            '''
            self.conn.execute(create_table_all_scores)


            # Create classification table
            create_table_category = '''
            CREATE TABLE IF NOT EXISTS all_categories (
                md5 TEXT NOT NULL PRIMARY KEY,
                category TEXT NOT NULL
            )
            '''
            self.conn.execute(create_table_category)

            
            # tag is the label for faces, multiple tags separated by commas indicate a group photo
            # UnmatchedFace found a face, but no match
            # NoFace no face found
            create_table_facetag = '''
            CREATE TABLE IF NOT EXISTS identified_faces (
                md5 TEXT NOT NULL PRIMARY KEY,
                tags TEXT NOT NULL
            )
            '''
            self.conn.execute(create_table_facetag)


            # Create a new face_identity_records table
            create_table_face_identity_records = '''
            CREATE TABLE IF NOT EXISTS face_identity_records  (
                face_md5 TEXT NOT NULL PRIMARY KEY,
                max_label TEXT NOT NULL,
                max_similarity REAL,
                file_md5 TEXT,
                left REAL DEFAULT 0,
                upper REAL DEFAULT 0,
                right REAL DEFAULT 0,
                lower REAL DEFAULT 0,
                is_trust INTEGER DEFAULT 0
            )
            '''
            self.conn.execute(create_table_face_identity_records)


            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")


        
    def save_md5(self, file_path, md5): 
        file_path = os.path.normpath(file_path)    

        try:# Delete all records with the same path but different MD5
            cur = self.conn.cursor()
            cur.execute('DELETE FROM photo_path WHERE file_path = ? and md5 != ?', (file_path, md5))
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")      

        try:           
            cur = self.conn.cursor()
            cur.execute('SELECT file_path FROM photo_path WHERE md5 = ?', (md5,))
            row = cur.fetchone()
            if row:
                if __debug__:
                    if row[0] != file_path: # If the path is different, it indicates an MD5 collision
                        self.logger.warning(f"md5 collision detected: {md5}, {file_path}, {row[0]}")
                cur.execute('REPLACE INTO photo_path (md5, file_path) VALUES (?, ?)', (md5, file_path))
                return False
            else:
                cur.execute('INSERT INTO photo_path (md5, file_path) VALUES (?, ?)', (md5, file_path))
                self.conn.commit()
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return False
        
    def batch_save_md5(self, md5_cache):
        try:
            cur = self.conn.cursor()
            self.conn.execute("BEGIN TRANSACTION;")
            for file_path, md5 in md5_cache.items():
                file_path = os.path.normpath(file_path)
                cur.execute('INSERT OR REPLACE INTO photo_path (md5, file_path) VALUES (?, ?)', (md5, file_path))
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            self.logger.error(f"Database error: {e}")          

    def get_file_path(self, md5):
        try:
            cursor = self.conn.execute('SELECT file_path FROM photo_path WHERE md5 = ?', (md5,))
            row = cursor.fetchone()
            return row[0] if row else None
            # for row in rows:
            #     if os.path.exists(row[0]):
            #         return row[0]
            # return None  
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None


    def load_md5_by_file_path(self, file_path):
        try:
            file_path = os.path.normpath(file_path)
            cursor = self.conn.execute('SELECT md5 FROM photo_path WHERE file_path = ?', (file_path,))
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None
        


    def save_photo_info(self, md5, photo_info):
        if photo_info is None or len(photo_info) == 0 or not isinstance(photo_info, dict):
            return
        try:
            # Save EXIF information
            cur = self.conn.cursor()

            # First, determine if it exists
            cur.execute('SELECT md5 FROM exif_data WHERE md5 = ?', (md5,))
            exists = cur.fetchone()

            if exists:
                # If it exists, update
                update_query = """
                UPDATE exif_data SET 
                    make = ?, 
                    model = ?, 
                    exposure_time = ?, 
                    f_number = ?, 
                    iso = ?, 
                    date_time_original = ?, 
                    focal_length = ?, 
                    shutter_speed_value = ?, 
                    aperture_value = ?, 
                    brightness_value = ?, 
                    flash = ?, 
                    white_balance = ?, 
                    metering_mode = ?, 
                    exposure_mode = ?, 
                    exposure_program = ?, 
                    gps_latitude = ?, 
                    gps_longitude = ?, 
                    gps_altitude = ?, 
                    lens_make = ?, 
                    lens_model = ? 
                WHERE md5 = ?"""
                cur.execute(update_query, (
                    photo_info.get('make'),
                    photo_info.get('model'),
                    photo_info.get('exposure_time'),
                    photo_info.get('f_number'),
                    photo_info.get('iso'),
                    photo_info.get('date_time_original'),
                    photo_info.get('focal_length'),
                    photo_info.get('shutter_speed_value'),
                    photo_info.get('aperture_value'),
                    photo_info.get('brightness_value'),
                    photo_info.get('flash'),
                    photo_info.get('white_balance'),
                    photo_info.get('metering_mode'),
                    photo_info.get('exposure_mode'),
                    photo_info.get('exposure_program'),
                    photo_info.get('gps_latitude'),
                    photo_info.get('gps_longitude'),
                    photo_info.get('gps_altitude'),
                    photo_info.get('lens_make'),
                    photo_info.get('lens_model'),
                    md5
                ))
            else:
                # If it does not exist, insert a new row
                insert_query = """
                INSERT INTO exif_data (md5, make, model, exposure_time, f_number, iso, 
                                    date_time_original, focal_length, shutter_speed_value, 
                                    aperture_value, brightness_value, flash, white_balance, 
                                    metering_mode, exposure_mode, exposure_program, 
                                    gps_latitude, gps_longitude, gps_altitude, lens_make, lens_model) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                cur.execute(insert_query, (
                    md5,
                    photo_info.get('make'),
                    photo_info.get('model'),
                    photo_info.get('exposure_time'),
                    photo_info.get('f_number'),
                    photo_info.get('iso'),
                    photo_info.get('date_time_original'),
                    photo_info.get('focal_length'),
                    photo_info.get('shutter_speed_value'),
                    photo_info.get('aperture_value'),
                    photo_info.get('brightness_value'),
                    photo_info.get('flash'),
                    photo_info.get('white_balance'),
                    photo_info.get('metering_mode'),
                    photo_info.get('exposure_mode'),
                    photo_info.get('exposure_program'),
                    photo_info.get('gps_latitude'),
                    photo_info.get('gps_longitude'),
                    photo_info.get('gps_altitude'),
                    photo_info.get('lens_make'),
                    photo_info.get('lens_model')
                ))

            self.conn.commit()
        except Exception as e:
            print(f"Error saving photo info: {e}")
            # Optionally roll back changes
            self.conn.rollback()

    def load_photo_info(self, md5):
        try:
            cursor = self.conn.execute('''SELECT md5,make,model,exposure_time,f_number,iso,date_time_original,
                                       focal_length,shutter_speed_value,aperture_value,brightness_value,flash,
                                       white_balance,metering_mode,exposure_mode,exposure_program,gps_latitude,
                                       gps_longitude,gps_altitude,lens_make,lens_model FROM exif_data WHERE md5 = ?''', (md5,))
            row = cursor.fetchone()
            if row:
                return {
                    'md5': row[0],
                    'make': row[1],
                    'model': row[2],
                    'exposure_time': row[3],
                    'f_number': row[4],
                    'iso': row[5],
                    'date_time_original': row[6],
                    'focal_length': row[7],
                    'shutter_speed_value': row[8],
                    'aperture_value': row[9],
                    'brightness_value': row[10],
                    'flash': row[11],
                    'white_balance': row[12],
                    'metering_mode': row[13],
                    'exposure_mode': row[14],
                    'exposure_program': row[15],
                    'gps_latitude': row[16],
                    'gps_longitude': row[17],
                    'gps_altitude': row[18],
                    'lens_make': row[19],
                    'lens_model': row[20]
                }
            return None
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None



    def load_photo_created_time(self, md5):
        try:
            cursor = self.conn.execute('SELECT date_time_original FROM exif_data WHERE md5 = ?', (md5,))
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None
        
    def batch_save_score(self, score_dict, metric_name):
        try:
            cur = self.conn.cursor()
            self.conn.execute("BEGIN TRANSACTION;")
            for md5, score in score_dict.items(): 
                cur.execute('INSERT OR REPLACE INTO all_scores (md5,metric_name,score) VALUES (?,?,?)', (md5,metric_name,score))                
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            self.logger.error(f"Database error: {e}")  

    def save_score(self, md5, score, metric_name):
        if score is None or score <= 0:
            return        
        try:
            self.conn.execute('INSERT OR REPLACE INTO all_scores (md5, metric_name, score) VALUES (?, ?, ?)',
                                (md5, metric_name, score))
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            

    def load_score(self, md5, metric_name):
        try:
            cursor = self.conn.execute('SELECT score FROM all_scores WHERE md5 = ? AND metric_name = ?', (md5, metric_name))            
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None

    def save_similarity(self, md5_1, md5_2, similarity):
        if similarity is None or similarity <= 0:
            return
        try:
            query = 'INSERT OR REPLACE INTO image_similarity (md5_1, md5_2, similarity) VALUES (?, ?, ?)'
            self.conn.execute(query, (md5_1, md5_2, similarity))
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")

    def load_similarity(self, md5_1, md5_2):
        try:
            # Combined query to consider two possible hash value combinations
            query = 'SELECT similarity FROM image_similarity WHERE (md5_1 = ? AND md5_2 = ?) OR (md5_1 = ? AND md5_2 = ?)'
            cursor = self.conn.execute(query, (md5_1, md5_2, md5_2, md5_1))
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None
        
    def batch_save_category(self, category_dict):
        try:
            cur = self.conn.cursor()
            self.conn.execute("BEGIN TRANSACTION;")
            for md5, category in category_dict.items(): 
                cur.execute('INSERT OR REPLACE INTO all_categories (md5,category) VALUES (?,?)', (md5,category))                
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            self.logger.error(f"Database error: {e}")

    def load_category(self, md5):
        try:
            cursor = self.conn.execute('SELECT category FROM all_categories WHERE md5 = ?', (md5,))
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None
               
    def set_face_labels(self,md5,tags):
        try:
            if isinstance(tags, str):
                tag = tags
            elif isinstance(tags, list):
                tag = ','.join(tags)
            else:
                raise Exception("tags must be FaceTag or slist type")
            
            # Check if the hash already exists
            cur = self.conn.cursor()
            cur.execute('SELECT md5 FROM identified_faces WHERE md5 = ?', (md5,))
            exists = cur.fetchone()

            if exists:
                # If it exists, updatetag值
                cur.execute('UPDATE identified_faces SET tags = ? WHERE md5 = ?', (tag, md5))
            else:
                # If it does not exist, insert a new row
                cur.execute('INSERT INTO identified_faces (md5, tags) VALUES (?, ?)', (md5, tag))

            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")

    
    def load_face_labels(self, md5):
        try:
            cursor = self.conn.execute('SELECT tags FROM identified_faces WHERE md5 = ?', (md5,))
            row = cursor.fetchone()
            return row[0].split(',') if row else None
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None
        
    # def has_confirmed_identity(self, md5):
    #     return self.load_face_labels(md5) is not None      
        
    def save_face_identity_record(self, face_md5, max_label, max_similarity, file_md5, box, is_trust):
        try:
            # First, search
            cur = self.conn.cursor()
            cur.execute('SELECT face_md5 FROM face_identity_records WHERE face_md5 = ?', (face_md5,))
            exists = cur.fetchone()
            if exists:
                # If it exists, update
                cur.execute('UPDATE face_identity_records SET max_label = ?, max_similarity = ?, file_md5 = ?, left = ?, upper = ?, right = ?, lower = ?, is_trust = ? WHERE face_md5 = ?', (max_label, max_similarity, file_md5, box[0], box[1], box[2], box[3], is_trust, face_md5))
            else:
                # If it does not exist, insert a new row
                cur.execute('INSERT INTO face_identity_records (face_md5, max_label, max_similarity, file_md5, left, upper, right, lower, is_trust) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (face_md5, max_label, max_similarity, file_md5, box[0], box[1], box[2], box[3], is_trust))
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")

    
    def load_face_identity_records(self, file_md5):
        try:
            cursor = self.conn.execute('SELECT face_md5, max_label, max_similarity, file_md5, left, upper, right, lower, is_trust FROM face_identity_records WHERE file_md5 = ?', (file_md5,))
            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({
                    'face_md5': row[0],
                    'max_label': row[1],
                    'max_similarity': row[2],
                    'file_md5': row[3],
                    'box': (row[4], row[5], row[6], row[7]),
                    'is_trust': row[8]
                })
            return result
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return None

        

if __name__ == "__main__":
    db_manager = DatabaseManager.get_instance()
    records = db_manager.load_face_identity_records('de496448640f23af5bf0bdda2d2f1145')
    print(records)