import unittest
import os
import json
from pathlib import Path
from youtube_semantic_segmenter import YoutubeSemanticSegmenter


class TestYoutubeSemanticSegmenter(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = "test_output"
        self.segmenter = YoutubeSemanticSegmenter(output_dir=self.test_output_dir)
        self.test_url = "https://www.youtube.com/watch?v=-uleG_Vecis"

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_output_dir):
            for root, dirs, files in os.walk(self.test_output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_output_dir)

    def test_download_video(self):
        """Test video download functionality"""
        audio_path = self.segmenter.download_video(self.test_url)
        self.assertTrue(os.path.exists(audio_path))
        self.assertTrue(audio_path.endswith(".wav"))

    def test_transcribe_audio(self):
        """Test audio transcription"""
        # First download the video
        audio_path = self.segmenter.download_video(self.test_url)

        # Then test transcription
        result = self.segmenter.transcribe_audio(audio_path)
        self.assertIsNotNone(result)
        self.assertIn("segments", result)
        self.assertIn("text", result)

    def test_create_semantic_chunks(self):
        """Test semantic chunking"""
        # Download and transcribe first
        audio_path = self.segmenter.download_video(self.test_url)
        transcription = self.segmenter.transcribe_audio(audio_path)

        # Test chunking
        chunks = self.segmenter.create_semantic_chunks(audio_path, transcription)
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 0)

        # Verify chunk properties
        for chunk in chunks:
            self.assertIn("start", chunk)
            self.assertIn("end", chunk)
            self.assertIn("text", chunk)
            self.assertIn("segments", chunk)
            # Check max duration constraint
            duration = chunk["end"] - chunk["start"]
            self.assertLessEqual(duration, 15000)  # 15 seconds in milliseconds

    def test_full_pipeline(self):
        """Test the entire pipeline"""
        result = self.segmenter.process_video(self.test_url)

        # Check result structure
        self.assertIn("audio_path", result)
        self.assertIn("num_chunks", result)
        self.assertIn("output_dir", result)

        # Check if files were created
        self.assertTrue(os.path.exists(result["audio_path"]))
        self.assertTrue(
            os.path.exists(os.path.join(result["output_dir"], "metadata.json"))
        )
        self.assertTrue(os.path.exists(os.path.join(result["output_dir"], "chunks")))

        # Check metadata
        with open(os.path.join(result["output_dir"], "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.assertEqual(len(metadata), result["num_chunks"])

            # Check chunk files
            for chunk in metadata:
                chunk_path = os.path.join(
                    result["output_dir"], "chunks", chunk["audio_file"]
                )
                self.assertTrue(os.path.exists(chunk_path))


if __name__ == "__main__":
    unittest.main()
