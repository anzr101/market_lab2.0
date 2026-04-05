"""
Checkpoint Manager - Save/Resume System
Ensures no work is lost if interrupted
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

class CheckpointManager:
    """
    Manages execution checkpoints for resumable processing
    Saves after each stock completes
    """
    
    def __init__(self, checkpoint_file: str = 'checkpoint.json'):
        self.checkpoint_file = Path(checkpoint_file)
        
    def save_progress(self, 
                     completed_stocks: List[str], 
                     remaining_stocks: List[str],
                     metadata: Optional[Dict] = None):
        """
        Save current progress
        
        Args:
            completed_stocks: List of completed stock tickers
            remaining_stocks: List of remaining stock tickers
            metadata: Optional additional info (model counts, errors, etc.)
        """
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'completed_stocks': completed_stocks,
            'remaining_stocks': remaining_stocks,
            'total_completed': len(completed_stocks),
            'total_remaining': len(remaining_stocks),
            'completion_percentage': len(completed_stocks) / (len(completed_stocks) + len(remaining_stocks)) * 100
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        # Write to file
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Print status
        total = len(completed_stocks) + len(remaining_stocks)
        pct = checkpoint['completion_percentage']
        print(f"\n💾 Checkpoint saved: {len(completed_stocks)}/{total} ({pct:.1f}%)")
        
    def load_checkpoint(self) -> Optional[Dict]:
        """
        Load checkpoint if exists
        
        Returns:
            Checkpoint dict or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            print("🆕 No checkpoint found - starting fresh")
            return None
        
        with open(self.checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        # Print resume info
        print("\n" + "="*80)
        print("🔄 RESUMING FROM CHECKPOINT")
        print("="*80)
        print(f"Last saved: {checkpoint['timestamp']}")
        print(f"Completed: {checkpoint['total_completed']} stocks")
        print(f"Remaining: {checkpoint['total_remaining']} stocks")
        print(f"Progress: {checkpoint['completion_percentage']:.1f}%")
        print("="*80)
        
        return checkpoint
    
    def clear_checkpoint(self):
        """Clear checkpoint file (call when batch complete)"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("🗑️  Checkpoint cleared (batch complete!)")
    
    def get_progress(self) -> float:
        """Get current completion percentage"""
        if not self.checkpoint_file.exists():
            return 0.0
        
        with open(self.checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        return checkpoint.get('completion_percentage', 0.0)


# Test the checkpoint manager
if __name__ == "__main__":
    print("Testing CheckpointManager...")
    
    mgr = CheckpointManager('test_checkpoint.json')
    
    # Simulate progress
    all_stocks = ['STOCK1', 'STOCK2', 'STOCK3', 'STOCK4', 'STOCK5']
    
    # First save
    mgr.save_progress(['STOCK1', 'STOCK2'], ['STOCK3', 'STOCK4', 'STOCK5'])
    
    # Load
    checkpoint = mgr.load_checkpoint()
    assert checkpoint['total_completed'] == 2
    assert checkpoint['total_remaining'] == 3
    
    # Continue
    mgr.save_progress(['STOCK1', 'STOCK2', 'STOCK3'], ['STOCK4', 'STOCK5'])
    
    # Complete
    mgr.save_progress(['STOCK1', 'STOCK2', 'STOCK3', 'STOCK4', 'STOCK5'], [])
    
    # Clear
    mgr.clear_checkpoint()
    
    # Verify cleared
    checkpoint = mgr.load_checkpoint()
    assert checkpoint is None
    
    print("✅ CheckpointManager working perfectly!")