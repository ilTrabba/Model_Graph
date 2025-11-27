import logging
from concurrent.futures import ThreadPoolExecutor

import requests
from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

class URLVerificationService:
    """Service for asynchronously verifying dataset URLs in the background."""
    
    def __init__(self, neo4j_service):
        self.neo4j_service = neo4j_service
        self.scheduler = BackgroundScheduler()
        self._is_running = False
        
    def verify_url(self, url: str, timeout: int = 10) -> bool:
        """
        Verify a URL by making an HTTP GET request.
        
        Args:
            url: The URL to verify
            timeout: Request timeout in seconds (default: 10)
            
        Returns:
            True if the URL returns HTTP 200 OK, False otherwise
        """
        try:
            response = requests.get(url, timeout=timeout, allow_redirects=True)
            return response.status_code == 200
        except requests.RequestException as e:
            logger.debug(f"URL verification failed for {url}: {e}")
            return False
    
    def _verify_and_update_model(self, model_id: str, dataset_url: str) -> None:
        """
        Verify a dataset URL and update the model in Neo4j.
        
        Args:
            model_id: The model ID to update
            dataset_url: The dataset URL to verify
        """
        try:
            verified = self.verify_url(dataset_url)
            self.neo4j_service.update_model(model_id, {'dataset_url_verified': verified})
            logger.info(f"Verified dataset URL for model {model_id}: {verified}")
        except Exception as e:
            logger.error(f"Failed to verify/update model {model_id}: {e}")
            # Mark as unverified on error
            try:
                self.neo4j_service.update_model(model_id, {'dataset_url_verified': False})
            except Exception:
                pass
    
    def verify_pending_datasets(self) -> None:
        """
        Query Neo4j for models with unverified dataset URLs and verify them.
        Processes up to 50 models per run using 5 parallel threads.
        """
        if not self.neo4j_service.is_connected():
            logger.warning("Neo4j not connected, skipping URL verification job")
            return
        
        try:
            # Query for models with unverified dataset URLs
            models = self._get_pending_models(limit=50)
            
            if not models:
                logger.debug("No pending dataset URLs to verify")
                return
            
            logger.info(f"Starting verification of {len(models)} dataset URLs")
            
            # Use ThreadPoolExecutor with 5 workers for parallel verification
            with ThreadPoolExecutor(max_workers=5) as executor:
                for model in models:
                    model_id = model.get('id')
                    dataset_url = model.get('dataset_url')
                    if model_id and dataset_url:
                        executor.submit(self._verify_and_update_model, model_id, dataset_url)
            
            logger.info(f"Completed verification batch of {len(models)} models")
            
        except Exception as e:
            logger.error(f"Error during dataset URL verification job: {e}")
    
    def _get_pending_models(self, limit: int = 50) -> list:
        """
        Get models with dataset URLs that haven't been verified yet.
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of model dictionaries with id and dataset_url
        """
        if not self.neo4j_service.driver:
            return []
        
        try:
            from src.config import Config
            with self.neo4j_service.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model)
                WHERE m.dataset_url IS NOT NULL 
                  AND m.dataset_url <> ''
                  AND m.dataset_url_verified IS NULL
                RETURN m.id as id, m.dataset_url as dataset_url
                LIMIT $limit
                """
                result = session.run(query, {'limit': limit})
                return [{'id': record['id'], 'dataset_url': record['dataset_url']} 
                        for record in result]
        except Exception as e:
            logger.error(f"Failed to query pending models: {e}")
            return []
    
    def start(self) -> None:
        """Start the background URL verification scheduler."""
        if self._is_running:
            logger.warning("URL verification scheduler is already running")
            return
        
        try:
            # Schedule the job to run every 5 minutes
            self.scheduler.add_job(
                self.verify_pending_datasets,
                'interval',
                minutes=5,
                id='url_verification_job',
                replace_existing=True
            )
            self.scheduler.start()
            self._is_running = True
            logger.info("URL verification scheduler started (runs every 5 minutes)")
        except Exception as e:
            logger.error(f"Failed to start URL verification scheduler: {e}")
    
    def stop(self) -> None:
        """Stop the background URL verification scheduler gracefully."""
        if not self._is_running:
            return
        
        try:
            self.scheduler.shutdown(wait=True)
            self._is_running = False
            logger.info("URL verification scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping URL verification scheduler: {e}")
