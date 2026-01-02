"""
File handling utilities for multi-file model uploads
Supports sharded safetensors validation and intelligent file sorting
"""

import os
import re
import torch
import zipfile
import tarfile
import logging
from typing import List, Tuple, Optional, Dict
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

class ShardedFileError(Exception):
    """Custom exception for sharded file validation errors"""
    pass

    def validate_sharded_safetensors(files_list: List[FileStorage]) -> Dict[str, any]:
        """
        Validates that uploaded files follow HuggingFace sharded pattern. 
        
        Pattern expected:  model-00001-of-00003.safetensors
        
        Validation rules:
        1. All files must match pattern: {base_name}-{shard_num}-of-{total_shards}.safetensors
        2. All files must have the same base_name and total_shards
        3. Shard numbers must be sequential:  1, 2, 3, ...  N (where N = total_shards)
        4. No duplicate shard numbers
        5. Number of files must equal total_shards
        
        Args:
            files_list: List of FileStorage objects from Flask request
            
        Returns: 
            Dict with validation info: 
            {
                'base_name': str,
                'total_shards': int,
                'shard_numbers': List[int],
                'sorted_files': List[FileStorage]  # sorted by shard number
            }
            
        Raises:
            ShardedFileError: If validation fails
        """
        if not files_list:
            raise ShardedFileError("No files provided")
        
        # Pattern regex:  capture base_name, shard_num, total_shards
        # Example: model-00001-of-00003.safetensors
        pattern = re.compile(r'^(. +? )-(\d+)-of-(\d+)\.safetensors$', re.IGNORECASE)
        
        parsed_files = []
        base_names = set()
        total_shards_values = set()
        shard_numbers = set()
        
        for file in files_list:
            filename = file.filename
            match = pattern.match(filename)
            
            if not match: 
                raise ShardedFileError(
                    f"File '{filename}' does not follow sharded pattern: "
                    f"expected format 'model-XXXXX-of-YYYYY.safetensors'"
                )
            
            base_name = match.group(1)
            shard_num = int(match.group(2))
            total_shards = int(match.group(3))
            
            base_names.add(base_name)
            total_shards_values.add(total_shards)
            
            # Check for duplicate shard numbers
            if shard_num in shard_numbers: 
                raise ShardedFileError(
                    f"Duplicate shard number {shard_num: 05d} found in uploaded files"
                )
            shard_numbers.add(shard_num)
            
            parsed_files.append({
                'file': file,
                'filename': filename,
                'base_name': base_name,
                'shard_num': shard_num,
                'total_shards': total_shards
            })
        
        # Validation 1: All files must have the same base_name
        if len(base_names) > 1:
            raise ShardedFileError(
                f"Inconsistent base names found: {base_names}. "
                f"All sharded files must belong to the same model."
            )
        
        # Validation 2: All files must declare the same total_shards
        if len(total_shards_values) > 1:
            raise ShardedFileError(
                f"Inconsistent total shard counts found: {total_shards_values}. "
                f"All files must declare the same 'of-XXXXX' value."
            )
        
        base_name = base_names.pop()
        total_shards = total_shards_values.pop()
        
        # Validation 3: Number of files must match declared total_shards
        if len(files_list) != total_shards:
            raise ShardedFileError(
                f"Expected {total_shards} files (from 'of-{total_shards: 05d}' pattern), "
                f"but received {len(files_list)} files"
            )
        
        # Validation 4: Shard numbers must be sequential (1, 2, 3, ...  N)
        expected_shards = set(range(1, total_shards + 1))
        if shard_numbers != expected_shards:
            missing = expected_shards - shard_numbers
            extra = shard_numbers - expected_shards
            
            error_msg = []
            if missing:
                missing_formatted = [f"{n:05d}" for n in sorted(missing)]
                error_msg.append(f"Missing shards: {', '.join(missing_formatted)}")
            if extra:
                extra_formatted = [f"{n:05d}" for n in sorted(extra)]
                error_msg.append(f"Unexpected shards: {', '.join(extra_formatted)}")
            
            raise ShardedFileError(
                f"Incomplete shard sequence.  {' '. join(error_msg)}"
            )
        
        # Sort files by shard number
        sorted_parsed = sorted(parsed_files, key=lambda x: x['shard_num'])
        sorted_files = [item['file'] for item in sorted_parsed]
        
        logger.info(
            f"✅ Sharded files validated:  {base_name} "
            f"({total_shards} shards, sequential 1-{total_shards})"
        )
        
        return {
            'base_name': base_name,
            'total_shards': total_shards,
            'shard_numbers': sorted([item['shard_num'] for item in sorted_parsed]),
            'sorted_files': sorted_files
        }


    def sort_sharded_files(file_paths: List[str]) -> List[str]:
        """
        Intelligently sort files based on their naming pattern.
        
        - If files follow sharded pattern (XXXXX-of-YYYYY), sort numerically by shard number
        - Otherwise, sort alphabetically
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Sorted list of file paths
        """
        # Pattern to detect sharded files
        shard_pattern = re.compile(r'-(\d+)-of-(\d+)\.(safetensors|bin|pt|pth)$', re.IGNORECASE)
        
        # Check if any file matches sharded pattern
        has_sharded = any(shard_pattern.search(os.path.basename(f)) for f in file_paths)
        
        if has_sharded: 
            # Sort by shard number
            def extract_shard_num(path):
                match = shard_pattern.search(os.path.basename(path))
                if match:
                    return int(match.group(1))
                # Non-sharded files go last
                return float('inf')
            
            sorted_files = sorted(file_paths, key=extract_shard_num)
            logger.debug(f"Sorted {len(file_paths)} files numerically by shard number")
            return sorted_files
        else: 
            # Alphabetical sort
            sorted_files = sorted(file_paths)
            logger.debug(f"Sorted {len(file_paths)} files alphabetically")
            return sorted_files


    def smart_load_bin(file_path: str, extract_dir: str) -> Tuple[bool, Optional[object]]:
        """
        Attempts to load a . bin file using multiple strategies.
        
        Strategy order:
        1. Direct torch.load() - standard PyTorch pickle
        2. Unzip - if it's a masked zip archive
        3. Untar - if it's a tarball
        4. Extract to directory and scan for model files
        
        Args:
            file_path: Path to . bin file
            extract_dir:  Directory for extraction if needed
            
        Returns: 
            Tuple (success:  bool, loaded_object: Optional[object])
            - If direct load succeeds: (True, torch_object)
            - If extraction succeeds: (True, None) - files extracted to extract_dir
            - If all fail: (False, None)
        """
        filename = os.path.basename(file_path)
        
        # Strategy 1: Direct torch.load()
        try:
            logger.info(f"[BIN LOAD] Attempting direct torch.load() on {filename}")
            loaded = torch.load(file_path, map_location="cpu")
            logger.info(f"✅ [BIN LOAD] Successfully loaded {filename} as PyTorch pickle")
            return (True, loaded)
        except Exception as e:
            logger.debug(f"[BIN LOAD] Direct load failed: {e}")
        
        # Strategy 2: Try unzip
        try:
            logger.info(f"[BIN LOAD] Attempting unzip on {filename}")
            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Check if files were extracted
            extracted_files = os.listdir(extract_dir)
            if extracted_files:
                logger.info(f"✅ [BIN LOAD] Extracted {len(extracted_files)} files from zip")
                return (True, None)
        except Exception as e:
            logger.debug(f"[BIN LOAD] Unzip failed:  {e}")
        
        # Strategy 3: Try untar
        try:
            logger.info(f"[BIN LOAD] Attempting untar on {filename}")
            with tarfile.open(file_path, 'r') as tf:
                tf.extractall(extract_dir)
            
            extracted_files = os.listdir(extract_dir)
            if extracted_files:
                logger.info(f"✅ [BIN LOAD] Extracted {len(extracted_files)} files from tarball")
                return (True, None)
        except Exception as e:
            logger.debug(f"[BIN LOAD] Untar failed: {e}")
        
        # All strategies failed
        logger.warning(f"❌ [BIN LOAD] Unable to load {filename} using any strategy")
        return (False, None)


    def scan_for_model_files(directory: str, include_no_extension: bool = True) -> List[str]:
        """
        Recursively scan directory for valid model files.
        
        Valid extensions: .bin, .pt, .pth, .ckpt, .safetensors
        Optionally includes files with no extension (validated with torch.load)
        
        Args:
            directory: Directory to scan
            include_no_extension:  If True, attempts to load files without extensions
            
        Returns:
            List of valid model file paths
        """
        valid_extensions = {'.bin', '.pt', '.pth', '.ckpt', '.safetensors'}
        found_files = []
        
        for root, _, files in os. walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                ext_lower = ext.lower()
                
                # Case 1: Has valid extension
                if ext_lower in valid_extensions:
                    found_files.append(file_path)
                    logger.debug(f"Found model file: {file}")
                
                # Case 2: No extension - try to validate as PyTorch binary
                elif ext == '' and include_no_extension: 
                    try:
                        # Attempt to load to verify it's a valid torch file
                        torch.load(file_path, map_location="cpu")
                        found_files.append(file_path)
                        logger.debug(f"Found extensionless model file (validated): {file}")
                    except Exception: 
                        # Not a valid torch file, skip
                        logger.debug(f"Skipping extensionless file (not torch): {file}")
                        continue
        
        logger.info(f"Scan complete:  found {len(found_files)} model files in {directory}")
        return found_files


    def is_likely_sharded_upload(files_list: List[FileStorage]) -> bool:
        """
        Quick check if uploaded files appear to be sharded safetensors.
        
        Args:
            files_list: List of uploaded files
            
        Returns: 
            True if files look like sharded safetensors (doesn't validate fully)
        """
        if len(files_list) <= 1:
            return False
        
        # Check if all are . safetensors
        if not all(f.filename.endswith('.safetensors') for f in files_list):
            return False
        
        # Check if at least one matches sharded pattern
        shard_pattern = re.compile(r'-\d+-of-\d+\.safetensors$', re.IGNORECASE)
        return any(shard_pattern.search(f.filename) for f in files_list)