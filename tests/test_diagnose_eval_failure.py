"""Tests for the diagnose_eval_failure script."""

import json
import os
import tempfile
import tarfile
import unittest
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from diagnose_eval_failure import EvaluationDiagnostics


def create_mock_evaluation_archive(temp_dir: Path, scenario: str = 'failed'):
    """Create a mock evaluation archive for testing.
    
    Args:
        temp_dir: Temporary directory to create archive in
        scenario: Type of scenario to create ('failed', 'success', 'partial')
    """
    # Create directory structure
    eval_dir = temp_dir / 'eval_outputs' / 'test' / 'eval_run'
    eval_dir.mkdir(parents=True)
    logs_dir = eval_dir / 'logs'
    logs_dir.mkdir()
    
    # Create output.report.json based on scenario
    if scenario == 'failed':
        report = {
            'total_instances': 16,
            'submitted_instances': 0,
            'completed_instances': 0,
            'resolved_instances': 0,
            'unresolved_instances': 0,
            'empty_patch_instances': 0,
            'error_instances': 0,
            'total_tests': 0,
            'total_passed_tests': 0,
            'completed_ids': [],
            'resolved_ids': [],
            'unresolved_ids': []
        }
    elif scenario == 'success':
        report = {
            'total_instances': 16,
            'submitted_instances': 16,
            'completed_instances': 16,
            'resolved_instances': 12,
            'unresolved_instances': 4,
            'empty_patch_instances': 0,
            'error_instances': 0,
            'total_tests': 100,
            'total_passed_tests': 75,
            'completed_ids': [f'instance_{i}' for i in range(16)],
            'resolved_ids': [f'instance_{i}' for i in range(12)],
            'unresolved_ids': [f'instance_{i}' for i in range(12, 16)]
        }
    else:  # partial
        report = {
            'total_instances': 16,
            'submitted_instances': 8,
            'completed_instances': 8,
            'resolved_instances': 4,
            'unresolved_instances': 4,
            'empty_patch_instances': 0,
            'error_instances': 0,
            'total_tests': 50,
            'total_passed_tests': 30,
            'completed_ids': [f'instance_{i}' for i in range(8)],
            'resolved_ids': [f'instance_{i}' for i in range(4)],
            'unresolved_ids': [f'instance_{i}' for i in range(4, 8)]
        }
    
    with open(eval_dir / 'output.report.json', 'w') as f:
        json.dump(report, f)
    
    # Create output.jsonl based on scenario
    if scenario == 'failed':
        # Empty file
        (eval_dir / 'output.jsonl').touch()
    else:
        # Create some output lines
        with open(eval_dir / 'output.jsonl', 'w') as f:
            num_lines = report['completed_instances']
            for i in range(num_lines):
                f.write(json.dumps({'instance_id': f'instance_{i}', 'result': 'resolved'}) + '\n')
    
    # Create log files based on scenario
    # Create log files matching total_instances for proper diagnosis
    num_instances = report['total_instances'] if scenario == 'failed' else 4
    for i in range(num_instances):
        log_file = logs_dir / f'instance_test{i}.log'
        
        if scenario == 'failed':
            log_content = f"""2026-03-10 21:44:18,067 - INFO - Starting instance test{i}
2026-03-10 21:46:38,177 - WARNING - [child] runtime init failure instance=test{i}
2026-03-10 21:46:38,177 - WARNING - [child] Instance test{i} failed: Conversation run failed for id=test-{i}: Remote conversation ended with error
"""
        else:
            log_content = f"""2026-03-10 21:44:18,067 - INFO - Starting instance test{i}
2026-03-10 21:46:38,177 - INFO - Instance test{i} completed successfully
"""
        
        with open(log_file, 'w') as f:
            f.write(log_content)
    
    # Create tar.gz archive
    archive_path = temp_dir / 'results.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(temp_dir / 'eval_outputs', arcname='eval_outputs')
    
    return archive_path


class TestEvaluationDiagnostics(unittest.TestCase):
    """Tests for the EvaluationDiagnostics class."""
    
    def test_analyze_failed_evaluation(self):
        """Test analysis of a completely failed evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = create_mock_evaluation_archive(temp_path, scenario='failed')
            
            diagnostics = EvaluationDiagnostics(str(archive_path))
            analysis = diagnostics.analyze()
            
            # Check report analysis
            self.assertEqual(analysis['report']['total_instances'], 16)
            self.assertEqual(analysis['report']['submitted_instances'], 0)
            self.assertEqual(analysis['report']['completed_instances'], 0)
            
            # Check output.jsonl analysis
            self.assertTrue(analysis['output_jsonl']['is_empty'])
            self.assertEqual(analysis['output_jsonl']['size'], 0)
            
            # Check logs analysis
            self.assertEqual(analysis['logs']['total_log_files'], 16)
            self.assertEqual(analysis['logs']['failure_patterns']['runtime_init_failure'], 16)
            self.assertEqual(analysis['logs']['failure_patterns']['remote_conversation_error'], 16)
            
            # Check diagnosis
            self.assertEqual(analysis['diagnosis']['severity'], 'CRITICAL')
            self.assertGreater(len(analysis['diagnosis']['issues']), 0)
            self.assertIn('Output.jsonl is empty', analysis['diagnosis']['issues'][0])
    
    def test_analyze_successful_evaluation(self):
        """Test analysis of a successful evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = create_mock_evaluation_archive(temp_path, scenario='success')
            
            diagnostics = EvaluationDiagnostics(str(archive_path))
            analysis = diagnostics.analyze()
            
            # Check report analysis
            self.assertEqual(analysis['report']['total_instances'], 16)
            self.assertEqual(analysis['report']['submitted_instances'], 16)
            self.assertEqual(analysis['report']['completed_instances'], 16)
            
            # Check output.jsonl analysis
            self.assertFalse(analysis['output_jsonl']['is_empty'])
            self.assertEqual(analysis['output_jsonl']['line_count'], 16)
            
            # Check diagnosis
            self.assertEqual(analysis['diagnosis']['severity'], 'INFO')
    
    def test_analyze_partial_failure(self):
        """Test analysis of a partially failed evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = create_mock_evaluation_archive(temp_path, scenario='partial')
            
            diagnostics = EvaluationDiagnostics(str(archive_path))
            analysis = diagnostics.analyze()
            
            # Check report analysis
            self.assertEqual(analysis['report']['total_instances'], 16)
            self.assertEqual(analysis['report']['submitted_instances'], 8)
            self.assertEqual(analysis['report']['completed_instances'], 8)
            
            # Check output.jsonl analysis
            self.assertFalse(analysis['output_jsonl']['is_empty'])
            self.assertEqual(analysis['output_jsonl']['line_count'], 8)
    
    def test_hypothesize_root_cause_llm_issue(self):
        """Test root cause hypothesis for LLM API issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = create_mock_evaluation_archive(temp_path, scenario='failed')
            
            diagnostics = EvaluationDiagnostics(str(archive_path))
            analysis = diagnostics.analyze()
            
            hypothesis = analysis['diagnosis']['root_cause_hypothesis']
            
            # Should identify it as an LLM API issue
            self.assertIn('LLM API issue', hypothesis)
            self.assertIn('infrastructure issue', hypothesis.lower())
            self.assertIn('Remote conversation ended with error', hypothesis)
    
    def test_failure_pattern_detection(self):
        """Test detection of specific failure patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = create_mock_evaluation_archive(temp_path, scenario='failed')
            
            diagnostics = EvaluationDiagnostics(str(archive_path))
            analysis = diagnostics.analyze()
            
            patterns = analysis['logs']['failure_patterns']
            
            # Should detect runtime init failures
            self.assertGreater(patterns['runtime_init_failure'], 0)
            # Should detect conversation failures
            self.assertGreater(patterns['conversation_run_failed'], 0)
            # Should detect remote conversation errors
            self.assertGreater(patterns['remote_conversation_error'], 0)


class TestDiagnoseEvalFailureIntegration(unittest.TestCase):
    """Integration test for the diagnose_eval_failure script."""
    
    def test_diagnose_eval_failure_script(self):
        """Test that we can create and analyze an archive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = create_mock_evaluation_archive(temp_path, scenario='failed')
            
            # Test that we can create and analyze an archive
            with EvaluationDiagnostics(str(archive_path)) as diagnostics:
                analysis = diagnostics.analyze()
                
                # Basic sanity checks
                self.assertIn('report', analysis)
                self.assertIn('output_jsonl', analysis)
                self.assertIn('logs', analysis)
                self.assertIn('diagnosis', analysis)
                
                # Should identify the critical failure
                self.assertEqual(analysis['diagnosis']['severity'], 'CRITICAL')


if __name__ == '__main__':
    unittest.main()
