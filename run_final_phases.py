"""
FINAL PHASES: 8 & 11 COMBINED
This script runs both phases sequentially - no need to wait!

Phase 8: Distribution Shift (~2-3 hours)
Phase 11: Continual Learning with EWC (~2-3 hours)
Total: ~4-6 hours

Just run this and go to sleep!
"""

import subprocess
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/final_phases.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_phase(phase_script, phase_name):
    """Run a phase script and wait for completion"""
    logger.info("=" * 100)
    logger.info(f"STARTING {phase_name}")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 100)
    
    try:
        # Run the phase script
        result = subprocess.run(
            ['python', phase_script],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {phase_name} COMPLETED SUCCESSFULLY!")
            return True
        else:
            logger.error(f"❌ {phase_name} FAILED with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {phase_name} FAILED with error: {e}")
        return False


def main():
    logger.info("\n" + "=" * 100)
    logger.info("FINAL PHASES: 8 & 11 - AUTOMATED RUN")
    logger.info("=" * 100)
    logger.info(f"Start Time: {datetime.now().isoformat()}")
    logger.info("This will run Phase 8, then Phase 11 automatically")
    logger.info("Estimated total time: 4-6 hours")
    logger.info("Go to sleep! Check back in the morning!")
    logger.info("=" * 100 + "\n")
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Phase 8: Distribution Shift
    phase8_success = run_phase('phase8_distribution_shift.py', 'PHASE 8: Distribution Shift')
    
    if not phase8_success:
        logger.error("Stopping - Phase 8 failed!")
        return
    
    logger.info("\n" + "🎉" * 40)
    logger.info("Phase 8 done! Moving to Phase 11...")
    logger.info("🎉" * 40 + "\n")
    
    # Phase 11: Continual Learning
    phase11_success = run_phase('phase11_continual_learning.py', 'PHASE 11: Continual Learning + EWC')
    
    if phase11_success:
        logger.info("\n" + "🎊" * 40)
        logger.info("ALL PHASES COMPLETED! YOU'RE DONE!")
        logger.info(f"Completion Time: {datetime.now().isoformat()}")
        logger.info("🎊" * 40 + "\n")
    else:
        logger.error("Phase 11 failed!")


if __name__ == "__main__":
    main()
