#!/usr/bin/env python3
"""Main synthetic data generation script for Gemma 30B.

Usage:
    python run_generation.py --config synthetic_generation/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kb_manager.config import load_config, PROJECT_ROOT
from kb_manager.models.database import Chunk
from kb_manager.web.app import db
from synthetic_generation.generators.qa_generator import QAGenerator
from synthetic_generation.generators.conv_generator import ConversationalGenerator
from synthetic_generation.generators.validator import SyntheticValidator
from kb_manager.llm import create_llm_client_from_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("synthetic_generation/output/generation.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


async def get_chunks_from_db(db_path: str, chunk_types: List[str] = None) -> List[Dict[str, Any]]:
    """Load chunks from database."""
    from sqlalchemy import select
    from kb_manager.models.database import Database, Chunk
    from kb_manager.config import DatabaseConfig

    if chunk_types is None:
        chunk_types = ["qa_pair"]

    db_cfg = DatabaseConfig(url_override=f"sqlite+aiosqlite:///{db_path}")
    database = Database(db_cfg)
    
    chunks_data = []
    async with database.session() as session:
        stmt = select(Chunk).where(Chunk.chunk_type.in_(chunk_types))
        result = await session.execute(stmt)
        chunks = result.scalars().all()
        
        for c in chunks:
            chunks_data.append({
                "id": c.id,
                "content": c.content,
                "chunk_type": c.chunk_type,
                "heading_path": c.heading_path,
                "document_id": c.document_id,
                "metadata": c.doc_metadata,
            })
    
    await database.close()
    return chunks_data


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_jsonl(data: List[Dict], path: Path) -> None:
    """Save data as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(data: List[Dict], path: Path) -> None:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Persian data for KB")
    parser.add_argument("--config", default="synthetic_generation/config.yaml", help="Config YAML path")
    parser.add_argument("--db-path", default="data/kb_test.db", help="SQLite DB path")
    parser.add_argument("--output-dir", default="synthetic_generation/output", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process")
    parser.add_argument("--skip-validation", action="store_true", help="Skip LLM validation")
    args = parser.parse_args()

    # Load config
    config = load_config_from_yaml(Path(args.config))
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create LLM client
    from kb_manager.config import load_config as load_app_config
    app_config = load_app_config()
    llm_client = create_llm_client_from_config(app_config)
    
    logger.info(f"Using LLM backend: {config.get('model', {}).get('backend', 'ollama')}")
    
    # Load chunks from DB
    chunk_types = config.get("database", {}).get("chunk_types", ["qa_pair"])
    logger.info(f"Loading chunks of types: {chunk_types}")
    chunks = await get_chunks_from_db(args.db_path, chunk_types)
    
    if args.limit:
        chunks = chunks[:args.limit]
    
    logger.info(f"Loaded {len(chunks)} chunks from database")
    
    # Build chunk text lookup for validation
    chunk_texts = {c["id"]: c["content"] for c in chunks}
    
    # Initialize generators
    qa_gen = QAGenerator(llm_client, config)
    conv_gen = ConversationalGenerator(llm_client, config)
    
    validator = None
    if not args.skip_validation:
        validator = SyntheticValidator(llm_client, config)
        logger.info(f"Validation enabled (threshold: {validator._threshold})")
    else:
        logger.info("Validation disabled")
    
    # Generation stats
    stats = {
        "total_chunks": len(chunks),
        "qa_generated": 0,
        "conversations_generated": 0,
        "qa_passed": 0,
        "conversations_passed": 0,
        "start_time": time.time(),
    }
    
    all_qa = []
    all_conversations = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["id"]
        content = chunk["content"]
        
        # Skip very short chunks
        min_len = config.get("generation", {}).get("min_chunk_length", 100)
        if len(content) < min_len:
            logger.debug(f"Skipping short chunk {chunk_id} ({len(content)} chars)")
            continue
        
        logger.info(f"[{i+1}/{len(chunks)}] Processing chunk {chunk_id[:8]}...")
        
        # Generate QA pairs
        try:
            qa_pairs = qa_gen.generate(content, chunk_id)
            stats["qa_generated"] += len(qa_pairs)
            
            # Validate if enabled
            if validator:
                valid_qa = validator.filter_batch(qa_pairs, chunk_texts)
                stats["qa_passed"] += len(valid_qa)
                qa_pairs = valid_qa
            
            all_qa.extend(qa_pairs)
            
        except Exception as e:
            logger.error(f"QA generation failed for chunk {chunk_id}: {e}")
        
        # Generate conversation (if ratio allows)
        import random
        conv_ratio = config.get("generation", {}).get("conversation_ratio", 0.3)
        if random.random() < conv_ratio:
            # Need a QA pair from chunk to base conversation on
            # Extract question/answer from chunk metadata
            fields = chunk.get("metadata", {}).get("fields", {})
            question = fields.get("question", "")
            answer = fields.get("answer", "") or fields.get("briefanswer", "")
            
            if question and answer:
                try:
                    conversation = conv_gen.generate(question, answer, chunk_id)
                    if conversation:
                        stats["conversations_generated"] += 1
                        
                        if validator:
                            valid = validator.validate_conversation(
                                conversation["conversation"], 
                                chunk_texts[chunk_id], 
                                chunk_id
                            )
                            if valid.passes_threshold:
                                stats["conversations_passed"] += 1
                                all_conversations.append(conversation)
                            else:
                                logger.debug(f"Conversation rejected: {valid.feedback}")
                        else:
                            all_conversations.append(conversation)
                except Exception as e:
                    logger.error(f"Conversation generation failed for chunk {chunk_id}: {e}")
        
        # Progress logging
        if (i + 1) % config.get("logging", {}).get("progress_interval", 100) == 0:
            elapsed = time.time() - stats["start_time"]
            logger.info(
                f"Progress: {i+1}/{len(chunks)} chunks | "
                f"QA: {stats['qa_generated']} generated, {stats['qa_passed']} passed | "
                f"Conv: {stats['conversations_generated']} generated, {stats['conversations_passed']} passed | "
                f"Elapsed: {elapsed:.1f}s"
            )
    
    # Save outputs
    output_format = config.get("output", {}).get("format", "jsonl")
    
    qa_path = output_dir / f"synthetic_qa.{output_format}"
    conv_path = output_dir / f"synthetic_conversations.{output_format}"
    
    if output_format == "jsonl":
        save_jsonl(all_qa, qa_path)
        save_jsonl(all_conversations, conv_path)
    else:
        save_json(all_qa, qa_path)
        save_json(all_conversations, conv_path)
    
    # Save metadata
    metadata = {
        "generation_config": config,
        "stats": stats,
        "elapsed_seconds": time.time() - stats["start_time"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(metadata, output_dir / "generation_metadata.json")
    
    # Validation stats
    if validator:
        val_stats = validator.get_statistics(
            [{"validation": q.get("validation", {})} for q in all_qa]
        )
        logger.info(f"Validation stats: {val_stats}")
    
    # Final summary
    elapsed = time.time() - stats["start_time"]
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total chunks processed: {stats['total_chunks']}")
    logger.info(f"QA pairs generated: {stats['qa_generated']}")
    logger.info(f"QA pairs passed validation: {stats['qa_passed']}")
    logger.info(f"Conversations generated: {stats['conversations_generated']}")
    logger.info(f"Conversations passed validation: {stats['conversations_passed']}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Output: {qa_path}, {conv_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())