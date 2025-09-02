#!/usr/bin/env python3
"""
CLI-based Graph-Driven Chatbot with Azure OpenAI integration
"""

import argparse
import json
import uuid
import os
import sys
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.runtime.graph_info import load_and_validate
from core.dst_manager import DSTManager

# API 키 로드
load_dotenv()

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('chatbot.log')
        ]
    )
    
    # OpenAI 클라이언트의 상세 디버그 로그 비활성화
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.INFO)


def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ env 파일 설정을 다시 확인:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n위 변수들 확인")
        return False
    
    return True

def print_session_info(session_info: Dict[str, Any]):
    """Print session information"""
    print(f"\n 세션 정보:")
    print(f"   Session ID: {session_info['session_id']}")
    print(f"   Current Node: {session_info.get('current_node', 'None')}")
    print(f"   Turn Count: {session_info.get('turn_count', 0)}")
    print(f"   Status: {'Complete' if session_info.get('is_complete') else 'Active'}")
    
    slots = session_info.get('slots', {})
    if slots:
        print(f"   Collected Slots:")
        for slot_name, slot_value in slots.items():
            print(f"     - {slot_name}: {slot_value}")

def validate_only(config_path: str, verbose: bool = False) -> bool:
    """Validate configuration without running chatbot"""
    try:
        print(f"Validating configuration: {config_path}")
        graph_info = load_and_validate(config_path)
        print(f"✅ Configuration validation completed successfully!")
        print(f"   Total nodes: {len(graph_info.nodes_info)}")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="그래프 기반 DST 챗봇",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli/chatbot.py --config config/sample_card_issuing.json
  
  # Validation only
  python cli/chatbot.py --config config/sample_card_issuing.json --validate-only
  
  # Advanced usage
  python cli/chatbot.py --config config/custom_flow.json --redis --session-id abc123 --start-node greeting
        """
    )
    
    parser.add_argument(
        '--config', 
        required=True, 
        help='Path to nodes JSON configuration file'
    )
    
    parser.add_argument(
        '--session-id', 
        help='Use specific session ID (default: generate new)'
    )
    
    parser.add_argument(
        '--redis', 
        action='store_true', 
        help='Use Redis for session storage (default: in-memory)'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--start-node', 
        help='Override start node (default: auto-detect)'
    )
    
    parser.add_argument(
        '--info', 
        action='store_true', 
        help='Show session info and exit'
    )
    
    parser.add_argument(
        '--validate-only', 
        action='store_true', 
        help='Validate configuration and exit (no chatbot execution)'
    )
    
    parser.add_argument(
        '--no-graph-validation', 
        action='store_true', 
        help='Disable enhanced graph validation (for faster loading)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle validation-only mode
    if args.validate_only:
        success = validate_only(args.config, args.verbose)
        sys.exit(0 if success else 1)
    
    # Validate environment (not needed for validation-only)
    if not validate_environment():
        sys.exit(1)
    
    try:
        # Load GraphRuntime once and reuse
        logger.info(f"Loading nodes from: {args.config}")
        graph_info = load_and_validate(args.config)
        logger.info(f"Loaded {len(graph_info.nodes_info)} nodes")
        
        # Initialize DST Manager (graph-driven)
        logger.info("Initializing DST Manager...")
        dst = DSTManager(graph_info=graph_info, use_redis=args.redis, start_node=args.start_node)
        
        # Handle session ID
        session_id = args.session_id or str(uuid.uuid4())
        
        # If info mode, just show session info
        # (get_session_info kept for backward compatibility)
        session_info = dst.get_session_info(session_id) if hasattr(dst, 'get_session_info') else None
        if args.info and session_info:
            print_session_info(session_info)
            return
        
        # Start or resume session
        if session_info:
            print(f"Resuming session: {session_id}")
            print_session_info(session_info)
        else:
            print(f"Starting new session: {session_id}")
            dst.start_session(session_id)
        
        print("\n" + "="*60)
        print("챗봇 시작")
        print("   명령어:")
        print("   - 'quit', 'exit', 'q': 종료")
        print("   - 'reset': 세션 리셋")
        print("   - 'info': 세션 정보 표시")
        print("="*60)
        print("안녕하세요! 무엇을 도와드릴까요?")

        # Main chat loop
        while True:
            try:
                print("\n" + "="*60)
                user_input = input("\n 사용자> ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n 채팅을 종료합니다.")
                    break
                
                elif user_input.lower() == 'reset':
                    dst.reset_session(session_id)
                    print("세션 리셋.")
                    continue
                
                elif user_input.lower() == 'info':
                    session_info = dst.get_session_info(session_id) if hasattr(dst, 'get_session_info') else None
                    if session_info:
                        print_session_info(session_info)
                    else:
                        print("세션 정보를 찾을 수 없습니다.")
                    continue
                
                # DST로 사용자 대화 처리
                dst_result = dst.process_turn(session_id, user_input)
                
                if dst_result.get('error'):
                    print(f"오류: {dst_result['response']}")
                    continue
                
                # 아웃풋 디스플레이
                print("\n" + "="*60)
                print(f"챗봇> {dst_result['response']}")
                
                # 디버깅 using verbose
                if args.verbose:
                    print(f"   [Debug] Node: {dst_result.get('current_node')}")
                    ctx = dst_result.get('context') or {}
                    print(f"   [Debug] NodeTurns: {ctx.get('node_turns')}")
                    print(f"   [Debug] TotalTurns: {ctx.get('total_turns')}")
                    print(f"   [Debug] Intent: {ctx.get('last_intent')}")
                    print(f"   [Debug] Context: {ctx}")
                    slots = dst_result.get('slots', {})
                    if slots:
                        print(f"   [Debug] Slots: {slots}")
                
                # 세션 완료 체킹
                if dst_result.get('session_complete'):
                    print("\n대화 완료")
                    restart = input("\n새로운 대화를 시작하시겠습니까? (y/n): ").strip().lower()
                    if restart in ['y', 'yes', '예', 'ㅇ', '네']:
                        session_id = str(uuid.uuid4())
                        dst.start_session(session_id)
                        print(f"새 세션 시작: {session_id}")
                    else:
                        break
                
            except KeyboardInterrupt:
                print("\n\n 채팅을 종료합니다.")
                break
            except EOFError:
                print("\n\n입력 종료.")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"예상치 못한 오류가 발생했습니다: {e}")
                print("대화를 계속하거나 'quit'으로 종료하세요.")
    
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON 파일 형식 오류: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        print(f"초기화 실패: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main() 