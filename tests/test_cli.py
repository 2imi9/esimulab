"""Tests for CLI entry point."""

from unittest.mock import patch

from click.testing import CliRunner

from esimulab.cli import main


class TestCli:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "bbox" in result.output
        assert "datetime" in result.output
        assert "steps" in result.output

    @patch("esimulab.pipeline.run_pipeline")
    def test_basic_invocation(self, mock_pipeline):
        runner = CliRunner()
        result = runner.invoke(main, ["--bbox", "-119.1,33.4,-118.9,35.4", "--no-gpu"])
        assert result.exit_code == 0
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["bbox"] == (-119.1, 33.4, -118.9, 35.4)
        assert call_kwargs["skip_gpu"] is True

    @patch("esimulab.pipeline.run_pipeline")
    def test_custom_steps(self, mock_pipeline):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--bbox", "-119.1,33.4,-118.9,35.4",
            "--steps", "100",
            "--no-gpu",
        ])
        assert result.exit_code == 0
        assert mock_pipeline.call_args[1]["num_steps"] == 100

    def test_bad_bbox(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--bbox", "not-a-bbox"])
        assert result.exit_code != 0

    @patch("esimulab.pipeline.run_pipeline")
    def test_datetime_parsing(self, mock_pipeline):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--bbox", "-119.1,33.4,-118.9,35.4",
            "--datetime", "2023-06-15T12:00:00",
            "--no-gpu",
        ])
        assert result.exit_code == 0
        call_time = mock_pipeline.call_args[1]["time"]
        assert call_time.year == 2023
        assert call_time.month == 6
