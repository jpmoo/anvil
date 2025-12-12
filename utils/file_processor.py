"""File processing utilities for training data upload"""

from pathlib import Path
from docx import Document
import json
from datetime import datetime
from utils.config import get_model_queue_dir

class FileProcessor:
    """Handles file upload and processing for training data"""
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.docx': 'docx'
    }
    
    @staticmethod
    def extract_text_from_docx(file_path: Path):
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error extracting DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: Path):
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""
    
    @classmethod
    def process_file(cls, uploaded_file, model_name: str):
        """Process uploaded file and save to queue directory"""
        
        # Get model-specific queue directory
        queue_dir = get_model_queue_dir(model_name)
        queue_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file type
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        if file_ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Save uploaded file
        file_path = queue_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text based on file type (for .txt and .docx)
        text = ""
        if file_ext == '.docx':
            text = cls.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            text = cls.extract_text_from_txt(file_path)
        # For .json and .jsonl, we don't extract text - they're already in the right format
        
        # Save metadata
        metadata = {
            "filename": uploaded_file.name,
            "date": datetime.now().isoformat(),
            "model": model_name,
            "file_type": cls.SUPPORTED_EXTENSIONS.get(file_ext, "unknown"),
            "text_length": len(text) if text else 0
        }
        
        metadata_path = queue_dir / f"{Path(uploaded_file.name).stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "success": True,
            "text": text,
            "metadata": metadata,
            "file_path": file_path
        }
    
    @staticmethod
    def delete_queue_file(filename: str, model_name: str):
        """Delete a queue file and its metadata"""
        try:
            queue_dir = get_model_queue_dir(model_name)
            filename_stem = Path(filename).stem
            
            # Delete the file
            file_path = queue_dir / filename
            deleted_files = []
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(filename)
            
            # Delete metadata
            metadata_path = queue_dir / f"{filename_stem}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
                deleted_files.append(metadata_path.name)
            
            if deleted_files:
                return {
                    "success": True, 
                    "message": f"Deleted {filename} and {len(deleted_files)} associated file(s)",
                    "deleted_files": deleted_files
                }
            else:
                return {"success": False, "error": "No files found to delete"}
        except Exception as e:
            return {"success": False, "error": str(e)}

