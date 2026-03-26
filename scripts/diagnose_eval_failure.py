#!/usr/bin/env python3
"""
Diagnostic script to analyze evaluation result archives and identify failures.

Usage:
    python scripts/diagnose_eval_failure.py <results_tar_gz_url_or_path>

Example:
    python scripts/diagnose_eval_failure.py https://results.eval.all-hands.dev/commit0/litellm_proxy-fireworks_ai-qwen3-coder-480b-a35b-instruct/22925541086/results.tar.gz
"""

import argparse
import json
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from urllib.request import urlretrieve


class EvaluationDiagnostics:
    """Diagnostic tool for analyzing evaluation failures."""

    def __init__(self, archive_path: str):
        self.archive_path = archive_path
        self.is_url = archive_path.startswith(('http://', 'https://'))
        self.temp_dir = None
        self.extract_dir = None

    def __enter__(self):
        if self.is_url:
            self.temp_dir = tempfile.mkdtemp()
            local_path = os.path.join(self.temp_dir, 'results.tar.gz')
            print(f"Downloading {self.archive_path}...")
            urlretrieve(self.archive_path, local_path)
            self.archive_path = local_path
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def analyze(self) -> Dict[str, Any]:
        """Analyze the evaluation results archive."""
        with tempfile.TemporaryDirectory() as extract_dir:
            self.extract_dir = extract_dir
            
            # Extract archive
            print(f"Extracting archive to {extract_dir}...")
            with tarfile.open(self.archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)

            # Find eval_outputs directory
            eval_outputs_path = self._find_eval_outputs(extract_dir)
            if not eval_outputs_path:
                return {'error': 'Could not find eval_outputs directory'}

            # Analyze report
            report_data = self._analyze_report(eval_outputs_path)
            
            # Analyze output.jsonl
            output_jsonl_data = self._analyze_output_jsonl(eval_outputs_path)
            
            # Analyze logs
            logs_data = self._analyze_logs(eval_outputs_path)

            return {
                'report': report_data,
                'output_jsonl': output_jsonl_data,
                'logs': logs_data,
                'diagnosis': self._generate_diagnosis(report_data, output_jsonl_data, logs_data)
            }

    def _find_eval_outputs(self, base_dir: str) -> Path:
        """Find the eval_outputs directory."""
        for root, dirs, files in os.walk(base_dir):
            if 'output.report.json' in files:
                return Path(root)
        return None

    def _analyze_report(self, eval_path: Path) -> Dict[str, Any]:
        """Analyze the output.report.json file."""
        report_path = eval_path / 'output.report.json'
        if not report_path.exists():
            return {'error': 'output.report.json not found'}

        with open(report_path) as f:
            report = json.load(f)

        return {
            'total_instances': report.get('total_instances', 0),
            'submitted_instances': report.get('submitted_instances', 0),
            'completed_instances': report.get('completed_instances', 0),
            'resolved_instances': report.get('resolved_instances', 0),
            'error_instances': report.get('error_instances', 0),
            'total_tests': report.get('total_tests', 0),
            'raw': report
        }

    def _analyze_output_jsonl(self, eval_path: Path) -> Dict[str, Any]:
        """Analyze the output.jsonl file."""
        output_path = eval_path / 'output.jsonl'
        if not output_path.exists():
            return {'error': 'output.jsonl not found'}

        file_size = output_path.stat().st_size
        
        if file_size == 0:
            return {
                'is_empty': True,
                'size': 0,
                'line_count': 0
            }

        with open(output_path) as f:
            lines = f.readlines()

        return {
            'is_empty': False,
            'size': file_size,
            'line_count': len(lines)
        }

    def _analyze_logs(self, eval_path: Path) -> Dict[str, Any]:
        """Analyze log files to identify failure patterns."""
        logs_dir = eval_path / 'logs'
        if not logs_dir.exists():
            return {'error': 'logs directory not found'}

        log_files = list(logs_dir.glob('instance_*.log'))
        
        failure_patterns = {
            'runtime_init_failure': 0,
            'conversation_run_failed': 0,
            'remote_conversation_error': 0,
            'timeout': 0,
            'other_errors': 0
        }
        
        error_messages = []
        instance_failures = {}

        for log_file in log_files:
            instance_name = log_file.stem.replace('instance_', '').replace('.log', '')
            
            with open(log_file) as f:
                content = f.read()

            # Check for specific error patterns
            if 'runtime init failure' in content:
                failure_patterns['runtime_init_failure'] += 1
            if 'Conversation run failed' in content:
                failure_patterns['conversation_run_failed'] += 1
            if 'Remote conversation ended with error' in content:
                failure_patterns['remote_conversation_error'] += 1
            if 'timeout' in content.lower():
                failure_patterns['timeout'] += 1

            # Extract error messages
            error_lines = [line for line in content.split('\n') if 'ERROR' in line or 'WARNING' in line]
            if error_lines:
                instance_failures[instance_name] = error_lines[-3:]  # Last 3 error/warning lines

            # Extract key error message
            match = re.search(r'error=([^\n]+)', content)
            if match and match.group(1) not in error_messages:
                error_messages.append(match.group(1))

        return {
            'total_log_files': len(log_files),
            'failure_patterns': failure_patterns,
            'unique_error_messages': error_messages,
            'sample_instance_failures': dict(list(instance_failures.items())[:3])  # Show first 3
        }

    def _generate_diagnosis(self, report: Dict, output_jsonl: Dict, logs: Dict) -> Dict[str, Any]:
        """Generate a diagnosis based on the analysis."""
        issues = []
        recommendations = []
        severity = 'INFO'

        # Check for empty output.jsonl
        if output_jsonl.get('is_empty', False):
            issues.append("Output.jsonl is empty - no instances completed successfully")
            severity = 'CRITICAL'

        # Check if any instances were submitted
        if report.get('submitted_instances', 0) == 0:
            issues.append(f"No instances were submitted out of {report.get('total_instances', 0)} total")
            severity = 'CRITICAL'
            
        # Check for runtime init failures
        if logs.get('failure_patterns', {}).get('runtime_init_failure', 0) > 0:
            count = logs['failure_patterns']['runtime_init_failure']
            issues.append(f"Runtime initialization failed for {count} instances")
            
        # Check for conversation failures
        if logs.get('failure_patterns', {}).get('remote_conversation_error', 0) > 0:
            count = logs['failure_patterns']['remote_conversation_error']
            issues.append(f"Remote conversation errors occurred in {count} instances")
            recommendations.append("Check LLM API configuration and connectivity")
            recommendations.append("Verify model availability and authentication")
            recommendations.append("Check for rate limiting or API quota issues")

        # All instances failed
        total = report.get('total_instances', 0)
        completed = report.get('completed_instances', 0)
        if total > 0 and completed == 0:
            issues.append("100% failure rate - all instances failed")
            recommendations.append("This suggests a systematic issue, not random failures")
            recommendations.append("Possible causes:")
            recommendations.append("  - LLM model API endpoint unavailable or misconfigured")
            recommendations.append("  - Authentication/authorization failure")
            recommendations.append("  - Model returning invalid response format")
            recommendations.append("  - Network connectivity issues")

        return {
            'severity': severity,
            'issues': issues,
            'recommendations': recommendations,
            'root_cause_hypothesis': self._hypothesize_root_cause(report, logs)
        }

    def _hypothesize_root_cause(self, report: Dict, logs: Dict) -> str:
        """Generate a hypothesis about the root cause."""
        remote_errors = logs.get('failure_patterns', {}).get('remote_conversation_error', 0)
        total = report.get('total_instances', 0)
        
        if remote_errors == total and total > 0:
            return (
                "All instances failed immediately with 'Remote conversation ended with error' "
                "after runtime initialization completed successfully. This strongly suggests an "
                "LLM API issue - the model endpoint may be unavailable, returning errors, or "
                "rejecting requests due to authentication/authorization problems. Since runtime "
                "setup (pod creation, health checks, repository cloning) completed normally, "
                "this is NOT an infrastructure issue."
            )
        
        return "Unable to determine root cause from available logs"


def print_report(analysis: Dict):
    """Print a formatted analysis report."""
    print("\n" + "="*80)
    print("EVALUATION FAILURE DIAGNOSTIC REPORT")
    print("="*80)

    # Report Summary
    print("\n📊 REPORT SUMMARY:")
    report = analysis.get('report', {})
    print(f"  Total Instances: {report.get('total_instances', 'N/A')}")
    print(f"  Submitted: {report.get('submitted_instances', 'N/A')}")
    print(f"  Completed: {report.get('completed_instances', 'N/A')}")
    print(f"  Resolved: {report.get('resolved_instances', 'N/A')}")
    print(f"  Errors: {report.get('error_instances', 'N/A')}")
    print(f"  Total Tests: {report.get('total_tests', 'N/A')}")

    # Output.jsonl Status
    print("\n📄 OUTPUT.JSONL STATUS:")
    output_jsonl = analysis.get('output_jsonl', {})
    if output_jsonl.get('is_empty', False):
        print("  ❌ EMPTY (0 bytes)")
    else:
        print(f"  ✓ Contains {output_jsonl.get('line_count', 0)} lines ({output_jsonl.get('size', 0)} bytes)")

    # Log Analysis
    print("\n📋 LOG ANALYSIS:")
    logs = analysis.get('logs', {})
    print(f"  Total Log Files: {logs.get('total_log_files', 0)}")
    print("  Failure Patterns:")
    for pattern, count in logs.get('failure_patterns', {}).items():
        if count > 0:
            print(f"    - {pattern}: {count}")

    # Error Messages
    errors = logs.get('unique_error_messages', [])
    if errors:
        print("  Unique Error Messages:")
        for error in errors[:5]:  # Show first 5
            print(f"    - {error}")

    # Diagnosis
    diagnosis = analysis.get('diagnosis', {})
    print(f"\n🔍 DIAGNOSIS (Severity: {diagnosis.get('severity', 'UNKNOWN')}):")
    
    issues = diagnosis.get('issues', [])
    if issues:
        print("  Issues Identified:")
        for issue in issues:
            print(f"    ❌ {issue}")
    
    recommendations = diagnosis.get('recommendations', [])
    if recommendations:
        print("\n  💡 Recommendations:")
        for rec in recommendations:
            print(f"    • {rec}")

    # Root Cause
    root_cause = diagnosis.get('root_cause_hypothesis', '')
    if root_cause:
        print("\n  🎯 Root Cause Hypothesis:")
        print(f"    {root_cause}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose evaluation failures from result archives'
    )
    parser.add_argument(
        'archive',
        help='Path or URL to results.tar.gz file'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()

    try:
        with EvaluationDiagnostics(args.archive) as diagnostics:
            analysis = diagnostics.analyze()
            
            if args.json:
                print(json.dumps(analysis, indent=2))
            else:
                print_report(analysis)
                
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
