#!/usr/bin/env elixir

# Profile Evaluator - Streams LinkedIn profiles and evaluates with AI APIs (Gemini, Claude, OpenAI)
# Usage: elixir evaluate_profiles.exs [model] [batch_size] [start_from] [mode] [stop_at]
# Modes: interactive (default), batch
# stop_at: Maximum number of profiles to process (default: all)

Mix.install(
  [
    {:req, "~> 0.5"},
    {:ex_rated, "~> 2.1"},
    {:dotenv, "~> 3.1"}
  ],
  config: [
    ex_rated: [
      timeout: 60_000,
      cleanup_rate: 10_000,
      persistent: false
    ]
  ]
)

defmodule ProgressTracker do
  @moduledoc """
  Simple progress tracking module for console output
  """

  defstruct [
    :model,
    :batch_size,
    :total_profiles,
    :start_time,
    current_batch: 0,
    profiles_processed: 0,
    requests_made: 0,
    accepted: 0,
    rejected: 0,
    errors: 0
  ]

  def new(model, batch_size, total_profiles) do
    %__MODULE__{
      model: model,
      batch_size: batch_size,
      total_profiles: total_profiles,
      start_time: DateTime.utc_now()
    }
  end

  def update(tracker, batch_num, batch_size, accepted, rejected) do
    new_tracker = %{
      tracker
      | current_batch: batch_num,
        profiles_processed: tracker.profiles_processed + batch_size,
        requests_made: tracker.requests_made + 1,
        accepted: tracker.accepted + accepted,
        rejected: tracker.rejected + rejected
    }

    display_progress(new_tracker)
    new_tracker
  end

  def add_error(tracker, _reason) do
    %{tracker | errors: tracker.errors + 1}
  end

  defp display_progress(tracker) do
    now = DateTime.utc_now()
    elapsed_seconds = DateTime.diff(now, tracker.start_time)

    # Calculate rates
    _rpm =
      if elapsed_seconds > 0, do: round(tracker.requests_made * 60 / elapsed_seconds), else: 0

    cpm =
      if elapsed_seconds > 0,
        do: round(tracker.profiles_processed * 60 / elapsed_seconds),
        else: 0

    # Calculate progress
    progress_percent =
      if tracker.total_profiles > 0 do
        round(tracker.profiles_processed * 100 / tracker.total_profiles)
      else
        0
      end

    # Calculate ETA
    remaining_profiles = tracker.total_profiles - tracker.profiles_processed

    eta =
      if cpm > 0 do
        remaining_minutes = remaining_profiles / cpm

        if remaining_minutes < 60 do
          "#{round(remaining_minutes)}m"
        else
          hours = div(round(remaining_minutes), 60)
          minutes = rem(round(remaining_minutes), 60)
          "#{hours}h #{minutes}m"
        end
      else
        "Calculating..."
      end

    # Create progress bar
    bar_width = 50
    filled_width = round(bar_width * progress_percent / 100)
    empty_width = bar_width - filled_width
    progress_bar = String.duplicate("█", filled_width) <> String.duplicate("░", empty_width)

    # Clear line and display progress
    # Clear current line
    IO.write("\r\e[K")

    IO.write(
      "🤖 Batch #{tracker.current_batch} | #{tracker.profiles_processed}/#{format_number(tracker.total_profiles)} (#{progress_percent}%) | "
    )

    IO.write("[#{progress_bar}] | ")
    IO.write("✅#{tracker.accepted} ❌#{tracker.rejected} | ")
    IO.write("#{cpm} candidates/min | ETA: #{eta}")
  end

  def format_number(num) when is_integer(num) and num >= 1000 do
    :erlang.float_to_binary(num / 1000, decimals: 1) <> "k"
  end

  def format_number(num), do: to_string(num)
end

defmodule ProfileEvaluator do
  @moduledoc """
  Evaluates LinkedIn profiles for ML architecture design capability using Gemini API
  with efficient streaming and rate limiting.
  """

  require Logger

  # Rate limits per minute for different models
  @rate_limits %{
    # Gemini models (Tier 1 API limits)
    "gemini-2.5-flash-lite" => %{rpm: 4000, optimal_batch: 10, provider: :gemini},
    "gemini-2.5-flash" => %{rpm: 1000, optimal_batch: 10, provider: :gemini},
    "gemini-2.5-pro" => %{rpm: 150, optimal_batch: 10, provider: :gemini},
    # Anthropic Claude models (Tier 4 rate limits)
    "claude-opus-4-1-20250805" => %{rpm: 4000, optimal_batch: 10, provider: :anthropic},
    "claude-opus-4-20250514" => %{rpm: 4000, optimal_batch: 10, provider: :anthropic},
    "claude-sonnet-4-20250514" => %{rpm: 4000, optimal_batch: 10, provider: :anthropic},
    "claude-3-7-sonnet-20250219" => %{rpm: 4000, optimal_batch: 10, provider: :anthropic},
    "claude-3-5-sonnet-20241022" => %{rpm: 4000, optimal_batch: 10, provider: :anthropic},
    "claude-3-5-haiku-20241022" => %{rpm: 4000, optimal_batch: 10, provider: :anthropic},
    "claude-3-5-sonnet-20240620" => %{rpm: 4000, optimal_batch: 10, provider: :anthropic},
    "claude-3-haiku-20240307" => %{rpm: 4000, optimal_batch: 10, provider: :anthropic},
    "claude-3-opus-20240229" => %{rpm: 4000, optimal_batch: 10, provider: :anthropic},
    # OpenAI GPT models (Tier 5 rate limits shown)
    "gpt-5" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-5-mini" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-5-nano" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4.1" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4.1-mini" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4.1-nano" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "o3" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "o4-mini" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4o" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4o-realtime-preview" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    # Specific model versions
    "gpt-4.1-2025-04-14" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4.1-mini-2025-04-14" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4.1-nano-2025-04-14" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4o-2024-05-13" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4o-2024-08-06" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4o-2024-11-20" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-4o-mini-2024-07-18" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-5-2025-08-07" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-5-mini-2025-08-07" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "gpt-5-nano-2025-08-07" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "o1-2024-12-17" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "o1-mini-2024-09-12" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "o1-pro-2025-03-19" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "o3-2025-04-16" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "o3-mini-2025-01-31" => %{rpm: 5000, optimal_batch: 20, provider: :openai},
    "o4-mini-2025-04-16" => %{rpm: 5000, optimal_batch: 20, provider: :openai}
  }

  @default_model "gemini-2.5-flash-lite"
  @input_file "data/Profiles/snap_me1d8qy32cqipopsau.jsonl.gz"
  @output_dir "data/out"

  def main(args \\ []) do
    # Load environment variables from .env file
    Dotenv.load()

    # Parse command-line options
    {options, remaining_args} = extract_options(args)

    # Store options in process dictionary for later use
    if options.prompt_file do
      Process.put(:prompt_file, options.prompt_file)
    end

    if options.output_file do
      Process.put(:output_file, options.output_file)
    end

    if options.input_file do
      Process.put(:input_file, options.input_file)
    end

    if options.input_profiles do
      Process.put(:input_profiles, options.input_profiles)
    end

    if options.model do
      Process.put(:model, options.model)
    end

    if options.batch_size do
      Process.put(:batch_size, options.batch_size)
    end

    if options.start_from do
      Process.put(:start_from, options.start_from)
    end

    if options.mode do
      Process.put(:mode, options.mode)
    end

    if options.stop_at do
      Process.put(:stop_at, options.stop_at)
    end

    if options.tsv_file do
      Process.put(:tsv_file, options.tsv_file)
    end

    if options.temperature do
      Process.put(:temperature, options.temperature)
    end

    if options.top_k do
      Process.put(:top_k, options.top_k)
    end

    if options.top_p do
      Process.put(:top_p, options.top_p)
    end

    if options.thinking_budget do
      Process.put(:thinking_budget, options.thinking_budget)
    end

    # Always store verbose level (defaults to 0)
    Process.put(:verbose, options.verbose)

    # Handle special commands
    case Enum.at(remaining_args, 0) do
      "help" ->
        show_usage()
        :ok

      "check_batch" ->
        batch_id = Enum.at(args, 1)

        if batch_id,
          do: check_batch_status(batch_id),
          else: IO.puts("Usage: elixir evaluate_profiles.exs check_batch <batch_id>")

        :ok

      "retrieve_batch" ->
        batch_id = Enum.at(args, 1)

        if batch_id,
          do: retrieve_batch_results(batch_id),
          else: IO.puts("Usage: elixir evaluate_profiles.exs retrieve_batch <batch_id>")

        :ok

      "check_batches" ->
        check_all_batches()
        :ok

      "retrieve_batches" ->
        retrieve_all_batches()
        :ok

      _ ->
        # Continue with normal processing
        run_evaluation(remaining_args)
    end
  end

  defp extract_options(args) do
    extract_options_recursive(
      args,
      %{
        prompt_file: nil,
        output_file: nil,
        input_file: nil,
        input_profiles: nil,
        model: nil,
        batch_size: nil,
        start_from: nil,
        mode: nil,
        stop_at: nil,
        tsv_file: nil,
        temperature: nil,
        top_k: nil,
        top_p: nil,
        thinking_budget: nil,
        verbose: 0
      },
      []
    )
  end

  defp extract_options_recursive([], options, remaining_args) do
    {options, Enum.reverse(remaining_args)}
  end

  defp extract_options_recursive([arg | rest], options, remaining_args) do
    cond do
      # Check for help flag
      arg in ["-h", "--help"] ->
        show_usage()
        exit({:shutdown, 0})

      # Check for verbose flag with multiple levels
      arg == "-v" ->
        extract_options_recursive(rest, %{options | verbose: options.verbose + 1}, remaining_args)

      arg == "--verbose" ->
        extract_options_recursive(
          rest,
          %{options | verbose: max(options.verbose, 1)},
          remaining_args
        )

      # Check for multiple v's (e.g., -vv, -vvv, -vvvv)
      String.starts_with?(arg, "-v") and not String.starts_with?(arg, "-vv") ->
        # This handles single -v which is already caught above
        extract_options_recursive(rest, options, [arg | remaining_args])

      String.match?(arg, ~r/^-v{2,}$/) ->
        # Count the number of v's
        v_count = String.length(arg) - 1
        extract_options_recursive(rest, %{options | verbose: v_count}, remaining_args)

      # Check for prompt file flag
      arg in ["-p", "--prompt"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | prompt_file: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for output file flag
      arg in ["-o", "--output"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | output_file: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for input file flag
      arg in ["-i", "--input"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | input_file: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for input profiles flag
      arg in ["--input_profiles"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | input_profiles: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for model flag
      arg in ["-m", "--model"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | model: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for batch size flag
      arg in ["-b", "--batch-size"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | batch_size: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for start from flag
      arg in ["-s", "--start-from"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | start_from: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for mode flag
      arg in ["--mode"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | mode: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for stop at flag
      arg in ["--stop-at"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | stop_at: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for TSV file flag
      arg in ["-t", "--tsv"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | tsv_file: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for temperature flag
      arg in ["--temperature"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | temperature: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for top-k flag
      arg in ["--top-k"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | top_k: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for top-p flag
      arg in ["--top-p"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(new_rest, %{options | top_p: value}, remaining_args)

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Check for thinking budget flag
      arg in ["--thinking-budget"] ->
        case rest do
          [value | new_rest] ->
            extract_options_recursive(
              new_rest,
              %{options | thinking_budget: value},
              remaining_args
            )

          [] ->
            IO.puts("❌ Error: #{arg} requires a value")
            exit({:shutdown, 1})
        end

      # Not an option, add to remaining args
      true ->
        extract_options_recursive(rest, options, [arg | remaining_args])
    end
  end

  defp show_usage do
    IO.puts("""
    🤖 Gemini Profile Evaluator

    Usage: elixir evaluate_profiles.exs -p prompt_file [OPTIONS]
           elixir evaluate_profiles.exs -h | --help

    Required Options:
      -p, --prompt    - Prompt file to use (e.g., evaluation_prompt_01.md or evaluation_prompt_02.md)

    Optional Options:
      -h, --help      - Show this help message
      -v, --verbose   - Increase output verbosity (can be repeated: -v, -vv, -vvv, -vvvv)
                        -v    : Show retry attempts and failures
                        -vv   : Also show when model provides non-JSON text
                        -vvv  : Also show full response from model
                        -vvvv : Also show full prompt sent to model
      -m, --model     - AI model to use (default: gemini-2.5-flash-lite)
                        Gemini models: gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro
                        Claude models: claude-opus-4-1-20250805, claude-sonnet-4-20250514,
                                      claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
                        OpenAI models: gpt-5-2025-08-07, gpt-4.1-2025-04-14, gpt-4o-2024-11-20,
                                      o1-2024-12-17, o3-2025-04-16, o4-mini-2025-04-16
      -i, --input     - Input file path (default: data/Profiles/snap_me1d8qy32cqipopsau.jsonl.gz)
      --input_profiles - Alternative input file path for profiles (overrides --input if provided)
      -o, --output    - Output file path (default: auto-generated in data/out/)
      -b, --batch-size - Number of profiles per API call (default: auto-optimal)
      -s, --start-from - Profile index to start from (default: 0)
      --mode          - Processing mode: interactive or batch (default: interactive)
      --stop-at       - Maximum profile index to process (default: all)
      -t, --tsv       - TSV file to process (enables TSV mode, forces interactive)

    Generation Config Options:
      --temperature   - Sampling temperature (0.0-1.0, default: 0.0 - deterministic)
      --top-k         - Top-K sampling (positive integer, default: disabled)
      --top-p         - Top-P sampling (0.0-1.0, default: 1.0 - no filtering)
      --thinking-budget - Thinking tokens budget (-1 for dynamic, 0 to disable, >0 for specific amount)

    Examples:
      # Process first 100 profiles with default settings
      elixir evaluate_profiles.exs -p evaluation_prompt_01.md -b 10 --stop-at 100

      # Process profiles with specific model and output file
      elixir evaluate_profiles.exs -p evaluation_prompt_02.md -m gemini-2.5-flash -o results.jsonl -b 10 --stop-at 100

      # Process profiles with custom generation config (more creative)
      elixir evaluate_profiles.exs -p evaluation_prompt_01.md --temperature 0.8 --top-p 0.9 --top-k 20 -b 10 --stop-at 100

      # Process profiles with thinking budget for reasoning (Gemini only)
      elixir evaluate_profiles.exs -p evaluation_prompt_02.md --thinking-budget 1024 -b 5 --stop-at 50

      # Process profiles with OpenAI GPT-4o
      elixir evaluate_profiles.exs -p evaluation_prompt_01.md -m gpt-4o-2024-11-20 -b 15 --stop-at 100

      # Process profiles with Claude Sonnet
      elixir evaluate_profiles.exs -p evaluation_prompt_02.md -m claude-3-5-sonnet-20241022 -b 20 --stop-at 50

      # Process profiles 1000-2000 in batch mode with custom input
      elixir evaluate_profiles.exs --prompt evaluation_prompt_01.md --input_profiles my_profiles.jsonl.gz --model gemini-2.5-flash -b 15 -s 1000 --mode batch --stop-at 2000

      # Process TSV file with profile summaries (TSV mode)
      elixir evaluate_profiles.exs -p evaluation_prompt_01.md --tsv profiles.tsv -b 5

      # Process TSV file with dynamic thinking and conservative sampling
      elixir evaluate_profiles.exs --prompt evaluation_prompt_02.md --model gemini-2.5-pro --output evaluated_profiles.tsv --tsv profiles.tsv --thinking-budget -1 --temperature 0.1

    TSV Mode:
      - Expects profile summaries in the first column
      - Preserves all existing columns in output
      - Adds evaluation column with model name as header
      - Only works in interactive mode (batch mode not supported)
      - Output: [original_columns] + [model_evaluation_column]

    Special Commands:
      elixir evaluate_profiles.exs help                  - Show this help message
      elixir evaluate_profiles.exs check_batch <id>      - Check batch status
      elixir evaluate_profiles.exs retrieve_batch <id>   - Retrieve batch results
      elixir evaluate_profiles.exs check_batches         - Check status of all batch jobs
      elixir evaluate_profiles.exs retrieve_batches      - Retrieve results from all completed batches
    """)
  end

  defp run_evaluation(args) do
    {model, batch_size, start_from, mode, stop_at, tsv_file} = parse_args(args)

    # Determine if we're in TSV mode
    if tsv_file do
      run_tsv_evaluation(model, batch_size, tsv_file)
    else
      run_profile_evaluation(model, batch_size, start_from, mode, stop_at)
    end
  end

  defp run_profile_evaluation(model, batch_size, start_from, mode, stop_at) do
    # Track start time
    start_time = System.monotonic_time(:millisecond)

    # Initialize counters
    Process.put(:total_input_tokens, 0)
    Process.put(:total_output_tokens, 0)
    Process.put(:total_profiles_processed, 0)

    # Ensure output directory exists
    File.mkdir_p!(@output_dir)

    # ExRated should start automatically with Mix.install application

    # Load evaluation prompt
    prompt = load_prompt()

    # Create output files
    timestamp = DateTime.utc_now() |> DateTime.to_unix()

    output_file =
      case Process.get(:output_file) do
        nil ->
          Path.join(
            @output_dir,
            "profile_evaluations_#{String.replace(model, ".", "_")}_#{timestamp}.jsonl"
          )

        custom_output ->
          # Use custom output file path
          custom_output
      end

    progress_file =
      Path.join(@output_dir, ".evaluation_progress_#{String.replace(model, ".", "_")}.txt")

    # Get total profile count (fast)
    actual_total = count_profiles()

    # Apply stop_at limit if specified
    profiles_to_process =
      if stop_at do
        min(stop_at - start_from, actual_total - start_from)
      else
        actual_total - start_from
      end

    total_profiles = start_from + profiles_to_process

    # Get rate limit config
    rate_config = Map.get(@rate_limits, model)
    if !rate_config, do: raise("Unknown model: #{model}")

    # Use optimal batch size if not specified
    batch_size = if batch_size == 0, do: rate_config.optimal_batch, else: batch_size

    # Initialize output file with metadata
    metadata = %{
      model: model,
      batch_size: batch_size,
      total_profiles: total_profiles,
      started_at: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    File.write!(output_file, JSON.encode!(%{metadata: metadata}) <> "\n")

    # Initialize progress tracker with the actual number of profiles to process
    progress_tracker = ProgressTracker.new(model, batch_size, profiles_to_process)

    # Display initial status
    IO.puts("🤖 Gemini Profile Evaluator")

    status_line =
      "Model: #{model} | Mode: #{String.upcase(to_string(mode))} | Batch Size: #{batch_size}"

    if stop_at do
      IO.puts(
        "#{status_line} | Processing: #{ProgressTracker.format_number(profiles_to_process)} profiles (#{start_from + 1} to #{total_profiles})"
      )
    else
      IO.puts("#{status_line} | Total Profiles: #{ProgressTracker.format_number(total_profiles)}")
    end

    case mode do
      :batch ->
        IO.puts("Batch Mode: 50% cost savings, ~24hr turnaround")

      :interactive ->
        IO.puts("Rate Limit: #{rate_config.rpm} requests/minute")
    end

    # Log generation parameters at verbosity level 1+
    verbose = Process.get(:verbose, 0)
    if verbose >= 1 do
      IO.puts("\n⚙️  Generation parameters:")
      
      # Check if model supports temperature
      model_lower = String.downcase(model)
      temperature_supported = not (String.contains?(model_lower, "gpt-5") or 
                                   String.contains?(model_lower, "o1") or
                                   String.starts_with?(model_lower, "o1-"))
      
      if temperature_supported do
        # Temperature (only if explicitly set)
        temp = Process.get(:temperature)
        if temp do
          IO.puts("   • Temperature: #{temp}")
        else
          IO.puts("   • Temperature: not set (using model default)")
        end
        
        # Top-P (only if explicitly set)
        top_p = Process.get(:top_p)
        if top_p do
          IO.puts("   • Top-P: #{top_p}")
        else
          IO.puts("   • Top-P: not set (using model default)")
        end
      else
        IO.puts("   • Temperature: not supported by #{model}")
        IO.puts("   • Top-P: not supported by #{model}")
      end
      
      # Top-K (Gemini only)
      if String.starts_with?(model, "gemini") do
        top_k = Process.get(:top_k)
        IO.puts("   • Top-K: #{top_k || "not set"}")
      end
      
      # Thinking budget (Gemini thinking models only)
      if String.contains?(model, "thinking") do
        thinking_budget = Process.get(:thinking_budget)
        IO.puts("   • Thinking budget: #{thinking_budget || "32768 (default)"}")
      end
    end
    
    IO.puts("\nStarting evaluation...\n")

    # Start processing
    config = %{
      model: model,
      batch_size: batch_size,
      start_from: start_from,
      stop_at: stop_at,
      prompt: prompt,
      output_file: output_file,
      progress_file: progress_file,
      total_profiles: total_profiles,
      profiles_to_process: profiles_to_process,
      rate_config: rate_config,
      progress_tracker: progress_tracker,
      mode: mode
    }

    case mode do
      :batch -> process_profiles_batch(config)
      :interactive -> process_profiles(config)
    end

    # Final newline after progress bar for interactive mode
    if mode == :interactive do
      IO.puts("")
      # Final summary
      show_summary(output_file)
      # Report token usage and runtime
      report_final_stats(start_time, model)
    end
  end

  defp parse_args(_args) do
    # All parameters now come from flags stored in process dictionary
    model = Process.get(:model, @default_model)

    batch_size =
      case Process.get(:batch_size) do
        # Will use optimal batch size
        nil -> 0
        size_str -> String.to_integer(size_str)
      end

    start_from =
      case Process.get(:start_from) do
        nil -> 0
        from_str -> String.to_integer(from_str)
      end

    mode =
      case Process.get(:mode) do
        "batch" -> :batch
        _ -> :interactive
      end

    stop_at =
      case Process.get(:stop_at) do
        # Process all profiles
        nil -> nil
        stop_str -> String.to_integer(stop_str)
      end

    tsv_file = Process.get(:tsv_file)

    {model, batch_size, start_from, mode, stop_at, tsv_file}
  end

  defp load_prompt do
    # Get prompt file from process dictionary (required)
    prompt_file = Process.get(:prompt_file)

    unless prompt_file do
      IO.puts("❌ Error: Prompt file is required. Use -p or --prompt to specify a prompt file.")
      IO.puts("Available prompt files: evaluation_prompt_01.md, evaluation_prompt_02.md")
      exit({:shutdown, 1})
    end

    case File.read(prompt_file) do
      {:ok, content} ->
        String.trim(content)

      {:error, reason} ->
        IO.puts("❌ Error: Cannot load prompt file #{prompt_file}: #{reason}")
        IO.puts("Please ensure the prompt file exists and is readable.")
        exit({:shutdown, 1})
    end
  end

  defp count_profiles do
    # Simple fallback: stream and count
    input_file = Process.get(:input_profiles) || Process.get(:input_file, @input_file)

    input_file
    |> File.stream!([:compressed])
    |> Enum.count()
  end

  defp process_profiles(config) do
    %{
      batch_size: batch_size
    } = config

    # Stream profiles and process in batches
    stream_profiles(config)
    |> Stream.chunk_every(batch_size)
    |> Stream.with_index(1)
    |> Enum.reduce(config, fn {batch, batch_num}, current_config ->
      process_batch(batch, batch_num, current_config)
    end)
  end

  defp process_batch(batch, batch_num, config) do
    %{
      model: model,
      prompt: prompt,
      output_file: output_file,
      progress_file: progress_file,
      rate_config: rate_config
    } = config

    Logger.info("Processing batch #{batch_num}...")

    # Format profiles for evaluation
    formatted_profiles = format_batch_profiles(batch)

    # Create evaluation prompt
    full_prompt = create_evaluation_prompt(formatted_profiles, prompt)

    # Wait for rate limit if needed
    bucket_name = "api_#{model}"
    # 1 minute in milliseconds
    time_scale = 60_000

    case ExRated.check_rate(bucket_name, time_scale, rate_config.rpm) do
      {:ok, _count} ->
        # Make API call with validation and retry
        case call_and_validate_response(batch, model, full_prompt) do
          {:ok, response} ->
            # Parse and save results
            {accepted, rejected} = save_batch_results(batch, batch_num, response, output_file)
            log_progress(batch_num, length(batch), progress_file)

            # Track total profiles processed
            current_total = Process.get(:total_profiles_processed, 0)
            Process.put(:total_profiles_processed, current_total + length(batch))

            # Update progress tracker
            updated_config =
              if Map.has_key?(config, :progress_tracker) do
                Map.put(
                  config,
                  :progress_tracker,
                  ProgressTracker.update(
                    config.progress_tracker,
                    batch_num,
                    length(batch),
                    accepted,
                    rejected
                  )
                )
              else
                config
              end

            updated_config

          {:error, reason} ->
            # Update progress tracker with error
            updated_config =
              if Map.has_key?(config, :progress_tracker) do
                Map.put(
                  config,
                  :progress_tracker,
                  ProgressTracker.add_error(config.progress_tracker, reason)
                )
              else
                config
              end

            updated_config
        end

      {:error, _limit} ->
        # Rate limited, wait and retry
        wait_time = calculate_wait_time(rate_config.rpm)
        Logger.info("Rate limited, waiting #{wait_time}ms...")
        Process.sleep(wait_time)
        process_batch(batch, batch_num, config)
    end
  end

  defp format_profile(profile) when is_map(profile) do
    # Extract key fields
    linkedin_id = profile["linkedin_id"] || profile["id"] || "unknown"

    full_name =
      profile["name"] ||
        "#{profile["first_name"] || ""} #{profile["last_name"] || ""}" ||
        "Unknown"

    headline = profile["position"] || ""
    summary = clean_text(profile["about"] || "", 300)
    experiences = format_experience(profile["experience"])
    education = format_education(profile["educations_details"] || profile["education"])
    certifications = format_certifications(profile["certifications"])

    """
    <Candidate>
    linkedin_id: #{linkedin_id}
    full_name: #{full_name}
    headline: #{headline}
    summary: #{summary}
    experiences: #{experiences}
    education: #{education}
    certifications: #{certifications}
    </Candidate>
    """
  end

  # Overload for TSV mode - format profile summary string
  defp format_profile({profile_summary, line_number}) when is_binary(profile_summary) do
    """
    <Candidate>
    linkedin_id: #{line_number}
      #{profile_summary}
    </Candidate>
    """
  end

  defp clean_text(text, max_length) when is_binary(text) do
    text
    |> String.replace(~r/[\n\r\t]/, " ")
    |> String.slice(0, max_length)
  end

  defp clean_text(_, _), do: ""

  defp format_experience(experiences) when is_list(experiences) do
    experiences
    |> Enum.map(fn exp ->
      title = exp["title"] || ""
      company = exp["company"] || ""
      dates = format_dates(exp)
      description = clean_text(exp["description"] || "", 200)

      base = "• #{title}, #{company}"
      base = if dates != "", do: base <> " (#{dates})", else: base
      if description != "", do: base <> ": #{description}", else: base
    end)
    |> Enum.join("; ")
  end

  defp format_experience(_), do: ""

  defp format_education(education) when is_list(education) do
    education
    |> Enum.map(fn edu ->
      field = edu["field_of_study"] || ""
      degree = edu["degree"] || ""
      school = edu["school"] || edu["school_name"] || ""
      dates = format_dates(edu)

      base = "#{field}, #{degree}, #{school}"
      if dates != "", do: base <> " (#{dates})", else: base
    end)
    |> Enum.join("; ")
  end

  defp format_education(_), do: ""

  defp format_certifications(certifications) when is_list(certifications) do
    certifications
    |> Enum.map(fn cert ->
      name = cert["name"] || ""
      authority = cert["authority"]

      if authority, do: "#{name} by #{authority}", else: name
    end)
    |> Enum.join("; ")
  end

  defp format_certifications(_), do: ""

  defp format_dates(%{"start_date" => start_date, "end_date" => end_date}) do
    start_str = format_single_date(start_date)
    end_str = format_single_date(end_date)

    cond do
      start_str != "" and end_str != "" -> "#{start_str} - #{end_str}"
      start_str != "" -> start_str
      end_str != "" -> end_str
      true -> ""
    end
  end

  defp format_dates(%{"meta" => meta}) when is_binary(meta), do: meta
  defp format_dates(%{"date" => date}) when is_binary(date), do: date
  defp format_dates(_), do: ""

  defp format_single_date(%{"year" => year, "month" => month, "day" => day}) do
    "#{year}-#{month}-#{day}"
  end

  defp format_single_date(%{"year" => year, "month" => month}) do
    "#{year}-#{month}"
  end

  defp format_single_date(%{"year" => year}) do
    "#{year}"
  end

  defp format_single_date(date) when is_binary(date), do: date
  defp format_single_date(_), do: ""

  defp create_evaluation_prompt(formatted_profiles, base_prompt) do
    """
    Evaluate these candidates for ML architecture design capability:

    #{formatted_profiles}

    #{base_prompt}

    Return your evaluation as a JSON object with an 'evaluations' array.
    Each evaluation should have 'linkedin_id' and 'evaluation' (ACCEPT or REJECT).

    Example response format:
    ```json
    {
      "evaluations": [
        {"linkedin_id": "john-doe-123", "evaluation": "ACCEPT"},
        {"linkedin_id": "jane-smith-456", "evaluation": "REJECT"}
      ]
    }
    ```
    """
  end

  defp format_batch_profiles(batch) do
    batch
    |> Enum.map(fn {profile, _index} -> format_profile(profile) end)
    |> Enum.join("\n\n")
  end

  defp stream_profiles(%{start_from: start_from, stop_at: stop_at}) do
    input_file = Process.get(:input_profiles) || Process.get(:input_file, @input_file)

    profile_stream =
      input_file
      |> File.stream!([:compressed])
      |> Stream.map(&String.trim/1)
      |> Stream.filter(&(&1 != ""))
      |> Stream.with_index()
      |> Stream.drop(start_from)

    # Apply stop_at limit if specified
    limited_stream =
      if stop_at do
        Stream.take(profile_stream, stop_at - start_from)
      else
        profile_stream
      end

    limited_stream
    |> Stream.map(fn {line, index} ->
      case JSON.decode(line) do
        {:ok, profile} -> {profile, index + 1}
        {:error, _} -> nil
      end
    end)
    |> Stream.filter(&(&1 != nil))
  end

  defp report_final_stats(start_time, model, profile_count \\ nil) do
    # Calculate runtime
    end_time = System.monotonic_time(:millisecond)
    runtime_ms = end_time - start_time
    runtime_seconds = runtime_ms / 1000.0

    # Get token usage
    input_tokens = Process.get(:total_input_tokens, 0)
    output_tokens = Process.get(:total_output_tokens, 0)
    total_tokens = input_tokens + output_tokens

    # Get profile count if not provided
    profile_count = profile_count || Process.get(:total_profiles_processed, 0)

    # Format runtime
    runtime_str =
      if runtime_seconds < 60 do
        "#{Float.round(runtime_seconds, 1)}s"
      else
        minutes = div(trunc(runtime_seconds), 60)
        seconds = rem(trunc(runtime_seconds), 60)
        "#{minutes}m #{seconds}s"
      end

    IO.puts("\n📈 Performance Statistics:")
    IO.puts("⏱️  Total runtime: #{runtime_str}")
    IO.puts("🎯 Model: #{model}")

    if total_tokens > 0 do
      IO.puts("🔤 Token usage:")
      IO.puts("   • Input tokens: #{format_number(input_tokens)}")
      IO.puts("   • Output tokens: #{format_number(output_tokens)}")
      IO.puts("   • Total tokens: #{format_number(total_tokens)}")

      # Calculate tokens per second if we have runtime
      if runtime_seconds > 0 do
        tokens_per_second = Float.round(total_tokens / runtime_seconds, 1)
        IO.puts("   • Processing rate: #{tokens_per_second} tokens/second")
      end

      # Calculate per-profile averages if we have profile count
      if profile_count > 0 do
        IO.puts("\n📊 Per-profile averages:")
        avg_input = Float.round(input_tokens / profile_count, 1)
        avg_output = Float.round(output_tokens / profile_count, 1)
        avg_total = Float.round(total_tokens / profile_count, 1)
        avg_time = Float.round(runtime_seconds / profile_count, 2)

        IO.puts("   • Input tokens: #{avg_input}")
        IO.puts("   • Output tokens: #{avg_output}")
        IO.puts("   • Total tokens: #{avg_total}")
        IO.puts("   • Processing time: #{avg_time}s")
      end
    end
  end

  defp format_number(num) when num < 1000, do: "#{num}"

  defp format_number(num) when num < 1_000_000 do
    "#{Float.round(num / 1000, 1)}K"
  end

  defp format_number(num) do
    "#{Float.round(num / 1_000_000, 2)}M"
  end

  defp update_token_usage(nil, model) do
    verbose = Process.get(:verbose, 0)
    if verbose >= 3 do
      IO.puts("⚠️  No usage data received for #{model}")
    end
    :ok
  end

  defp update_token_usage(usage, model) do
    # Get or initialize token counters
    input_tokens = Process.get(:total_input_tokens, 0)
    output_tokens = Process.get(:total_output_tokens, 0)
    
    verbose = Process.get(:verbose, 0)
    
    # Debug log the usage structure for GPT-5 models
    if verbose >= 3 and String.starts_with?(model, "gpt-5") do
      IO.puts("🔍 Token usage structure for #{model}: #{inspect(usage)}")
    end

    # Extract tokens based on provider format
    {new_input, new_output} =
      cond do
        # Anthropic and newer OpenAI format
        Map.has_key?(usage, "input_tokens") ->
          {usage["input_tokens"] || 0, usage["output_tokens"] || 0}
        
        # GPT-5 and older OpenAI format
        Map.has_key?(usage, "prompt_tokens") ->
          {usage["prompt_tokens"] || 0, usage["completion_tokens"] || 0}

        # Gemini format
        Map.has_key?(usage, "promptTokenCount") ->
          {usage["promptTokenCount"] || 0, usage["candidatesTokenCount"] || 0}

        true ->
          if verbose >= 2 do
            IO.puts("⚠️  Unknown usage format for #{model}: #{inspect(Map.keys(usage))}")
          end
          {0, 0}
      end

    # Update counters
    Process.put(:total_input_tokens, input_tokens + new_input)
    Process.put(:total_output_tokens, output_tokens + new_output)

    # Track per-model usage if verbose
    if verbose >= 2 do
      IO.puts("📊 Token usage: +#{new_input} input, +#{new_output} output")
    end
  end

  defp call_and_validate_response(batch, model, prompt, retry_count \\ 0) do
    max_retries = 3

    case call_api(model, prompt) do
      {:ok, %{text: response, usage: usage}} ->
        # Store usage for reporting
        update_token_usage(usage, model)

        # Try to parse as JSON first (structured output)
        parsed_response =
          case JSON.decode(response) do
            {:ok, json} when is_map(json) ->
              # Convert structured response to expected format
              if evaluations = json["evaluations"] do
                evaluations
                |> Enum.map(fn eval ->
                  "#{eval["linkedin_id"]}: #{eval["evaluation"]}"
                end)
                |> Enum.join("\n")
              else
                # Fall back to text parsing
                response
              end

            _ ->
              # Use original text response
              response
          end

        # Validate response format
        case validate_response(batch, parsed_response) do
          {:ok, validated_response} ->
            {:ok, validated_response}

          {:error, validation_error} ->
            if retry_count < max_retries do
              Logger.warning(
                "Response validation failed (retry #{retry_count + 1}/#{max_retries}): #{validation_error}"
              )

              Process.sleep(1000)
              # On retry, try without structured output if first attempt used it
              call_and_validate_response(batch, model, prompt, retry_count + 1)
            else
              Logger.error(
                "Response validation failed after #{max_retries} retries: #{validation_error}"
              )

              {:error, "Validation failed: #{validation_error}"}
            end
        end

      {:error, reason} ->
        if retry_count < max_retries do
          Logger.warning("API call failed (retry #{retry_count + 1}/#{max_retries}): #{reason}")
          Process.sleep(2000)
          call_and_validate_response(batch, model, prompt, retry_count + 1)
        else
          {:error, reason}
        end
    end
  end

  defp validate_response(batch, response) do
    expected_count = length(batch)

    # Parse response lines looking for the specific format [linkedin_id]: ACCEPT/REJECT
    response_lines =
      response
      |> String.split("\n")
      |> Enum.map(&String.trim/1)
      |> Enum.filter(&(&1 != ""))

    # Look for lines that match the expected pattern: [something]: ACCEPT or [something]: REJECT
    valid_results =
      response_lines
      |> Enum.map(fn line ->
        # Match pattern: [linkedin_id]: ACCEPT/REJECT
        case Regex.run(~r/^\[([^\]]+)\]\s*:\s*(ACCEPT|REJECT)\s*$/i, line) do
          [_, linkedin_id, evaluation] ->
            %{linkedin_id: String.trim(linkedin_id), evaluation: String.upcase(evaluation)}

          _ ->
            # Try alternative format: linkedin_id: ACCEPT/REJECT (without brackets)
            case String.split(line, ":", parts: 2) do
              [id_part, eval_part] ->
                cleaned_id = String.trim(id_part) |> String.replace(~r/^\[|\]$/, "")
                cleaned_eval = String.trim(eval_part) |> String.upcase()

                if cleaned_eval in ["ACCEPT", "REJECT"] and not String.contains?(cleaned_id, " ") do
                  %{linkedin_id: cleaned_id, evaluation: cleaned_eval}
                else
                  nil
                end

              _ ->
                nil
            end
        end
      end)
      |> Enum.filter(&(&1 != nil))

    if length(valid_results) == expected_count do
      {:ok,
       Enum.map(valid_results, fn result ->
         "#{result.linkedin_id}: #{result.evaluation}"
       end)
       |> Enum.join("\n")}
    else
      {:error,
       "Expected #{expected_count} valid results, got #{length(valid_results)}. Found valid: #{inspect(valid_results)}"}
    end
  end

  defp call_api(model, prompt) do
    rate_config = Map.get(@rate_limits, model)

    unless rate_config do
      raise("Unsupported model: #{model}")
    end

    case rate_config.provider do
      :gemini -> call_gemini_api(model, prompt)
      :anthropic -> call_anthropic_api(model, prompt)
      :openai -> call_openai_api(model, prompt)
    end
  end

  defp call_anthropic_api(model, prompt) do
    api_key = System.get_env("ANTHROPIC_API_KEY")
    if !api_key, do: raise("ANTHROPIC_API_KEY environment variable not set")

    url = "https://api.anthropic.com/v1/messages"

    # Build payload for Messages API with assistant prefill
    payload = %{
      "model" => model,
      "max_tokens" => 4096,
      "messages" => [
        %{
          "role" => "user",
          "content" => prompt
        },
        %{
          "role" => "assistant",
          "content" => "```json\n{\n  \"evaluations\": [\n    {\"linkedin_id\": \""
        }
      ]
    }

    # Add generation config if provided
    payload = add_anthropic_generation_config(payload)

    headers = [
      {"x-api-key", api_key},
      {"anthropic-version", "2023-06-01"},
      {"Content-Type", "application/json"}
    ]

    case Req.post(url, json: payload, headers: headers, receive_timeout: 300_000) do
      {:ok, %{status: 200, body: body}} ->
        # Extract text from Anthropic response
        text =
          case body["content"] do
            [%{"text" => text} | _] ->
              # Prepend the prefilled content to complete the JSON
              "```json\n{\n  \"evaluations\": [\n    {\"linkedin_id\": \"" <> text

            _ ->
              ""
          end

        # Extract token usage
        usage = body["usage"]

        {:ok, %{text: text || "", usage: usage}}

      {:ok, %{status: status, body: body}} ->
        {:error, "HTTP #{status}: #{inspect(body)}"}

      {:error, reason} ->
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  defp add_anthropic_generation_config(payload) do
    # Get model name from payload to check if it's Opus
    model = Map.get(payload, "model", "")
    is_opus = String.contains?(String.downcase(model), "opus")

    # Start with no default parameters
    config_updates = %{}

    # Temperature - only add if user specified
    config_updates =
      case Process.get(:temperature) do
        nil -> config_updates
        temp_str -> Map.put(config_updates, "temperature", parse_float(temp_str))
      end

    # Top-P - only add if user explicitly specified (skip for Opus if temperature is set)
    config_updates =
      case Process.get(:top_p) do
        nil ->
          # Don't add top_p unless explicitly specified
          config_updates

        top_p_str ->
          if is_opus and Map.has_key?(config_updates, "temperature") do
            # For Opus, if temperature is set, don't add top_p
            IO.puts(
              "⚠️  Note: Opus models don't support both temperature and top_p. Using temperature only."
            )

            config_updates
          else
            Map.put(config_updates, "top_p", parse_float(top_p_str))
          end
      end

    # Top-K - only add if user specified (no default)
    config_updates =
      case Process.get(:top_k) do
        nil -> config_updates
        top_k_str -> Map.put(config_updates, "top_k", parse_integer(top_k_str))
      end

    # Note: Anthropic doesn't support thinking budget - that's a Gemini feature
    # We'll ignore thinking_budget for Anthropic models

    Map.merge(payload, config_updates)
  end

  defp call_openai_api(model, prompt) do
    api_key = System.get_env("OPENAI_API_KEY")
    if !api_key, do: raise("OPENAI_API_KEY environment variable not set")

    url = "https://api.openai.com/v1/chat/completions"

    # Build payload for Chat Completions API
    payload = %{
      "model" => model,
      "messages" => [
        %{
          "role" => "user",
          "content" => prompt
        }
      ],
      "max_completion_tokens" => 4096
    }

    # Add generation config if provided
    payload = add_openai_generation_config(payload)

    headers = [
      {"Authorization", "Bearer #{api_key}"},
      {"Content-Type", "application/json"}
    ]

    case Req.post(url, json: payload, headers: headers, receive_timeout: 300_000) do
      {:ok, %{status: 200, body: body}} ->
        # Extract text from OpenAI response (handles both standard and GPT-5 formats)
        text =
          cond do
            # Standard OpenAI format
            body["choices"] ->
              case body["choices"] do
                [%{"message" => %{"content" => content}} | _] -> content
                _ -> ""
              end
            
            # GPT-5 format with output array
            body["output"] ->
              case body["output"] do
                [%{"content" => [%{"text" => text} | _]} | _] -> text
                _ -> ""
              end
            
            true -> ""
          end

        # Extract token usage (same location in both formats)
        usage = body["usage"]
        
        # Debug log for GPT-5 models
        verbose = Process.get(:verbose, 0)
        if verbose >= 3 and String.starts_with?(model, "gpt-5") do
          IO.puts("🔍 GPT-5 Response structure: #{inspect(Map.keys(body))}")
          if usage do
            input_key = if Map.has_key?(usage, "input_tokens"), do: "input_tokens", else: "prompt_tokens"
            output_key = if Map.has_key?(usage, "output_tokens"), do: "output_tokens", else: "completion_tokens"
            IO.puts("   Usage found: input=#{usage[input_key]}, output=#{usage[output_key]}")
          else
            IO.puts("   No usage data found in response")
          end
        end

        {:ok, %{text: text || "", usage: usage}}

      {:ok, %{status: status, body: body}} ->
        {:error, "HTTP #{status}: #{inspect(body)}"}

      {:error, reason} ->
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  defp add_openai_generation_config(payload) do
    model = Map.get(payload, "model", "")
    model_lower = String.downcase(model)

    # GPT-5 and o1 models don't support temperature/top_p settings
    is_restricted_model =
      String.contains?(model_lower, "gpt-5") or
        String.contains?(model_lower, "o1") or
        String.starts_with?(model_lower, "o1-")

    if is_restricted_model do
      # Don't add temperature or top_p for restricted models
      payload
    else
      # Start with no default parameters
      config_updates = %{}

      # Temperature - only add if user specified
      config_updates =
        case Process.get(:temperature) do
          nil -> config_updates
          temp_str -> Map.put(config_updates, "temperature", parse_float(temp_str))
        end

      # Top-P - only add if user explicitly specified
      config_updates =
        case Process.get(:top_p) do
          nil -> 
            # Don't add top_p unless explicitly specified
            config_updates
          top_p_str -> 
            Map.put(config_updates, "top_p", parse_float(top_p_str))
        end

      # Note: OpenAI doesn't support top_k in the same way as Gemini
      # We'll ignore top_k for OpenAI models

      # Note: OpenAI doesn't support thinking budget - that's a Gemini feature
      # We'll ignore thinking_budget for OpenAI models

      Map.merge(payload, config_updates)
    end
  end

  defp call_gemini_api(model, prompt) do
    api_key = System.get_env("GEMINI_API_KEY")
    if !api_key, do: raise("GEMINI_API_KEY environment variable not set")

    url = "https://generativelanguage.googleapis.com/v1beta/models/#{model}:generateContent"

    # Build base payload
    base_payload = %{
      "contents" => [
        %{
          "parts" => [
            %{"text" => prompt}
          ]
        }
      ]
    }

    # Add structured output configuration with optional generation parameters
    generation_config = build_generation_config()
    payload = Map.put(base_payload, "generationConfig", generation_config)

    headers = [
      {"x-goog-api-key", api_key},
      {"Content-Type", "application/json"}
    ]

    # Increase timeout for thinking models (5 minutes)
    case Req.post(url, json: payload, headers: headers, receive_timeout: 300_000) do
      {:ok, %{status: 200, body: body}} ->
        # Extract text from Gemini response
        text =
          get_in(body, ["candidates", Access.at(0), "content", "parts", Access.at(0), "text"])

        # Extract token usage
        usage = body["usageMetadata"]

        {:ok, %{text: text || "", usage: usage}}

      {:ok, %{status: status, body: body}} ->
        {:error, "HTTP #{status}: #{inspect(body)}"}

      {:error, reason} ->
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  defp build_generation_config do
    # Start with structured output configuration only
    base_config = %{
      "responseMimeType" => "application/json",
      "responseSchema" => evaluation_response_schema()
      # No default temperature, topP, or topK
    }

    # Override with user-provided parameters if present
    config =
      base_config
      |> add_if_present("temperature", Process.get(:temperature), &parse_float/1)
      |> add_if_present("topK", Process.get(:top_k), &parse_integer/1)
      |> add_if_present("topP", Process.get(:top_p), &parse_float/1)

    # Handle thinking budget specially - it has special values
    case Process.get(:thinking_budget) do
      nil -> config
      value -> add_thinking_budget(config, value)
    end
  end

  defp add_if_present(config, _key, nil, _parser), do: config

  defp add_if_present(config, key, value, parser) when is_binary(value) do
    Map.put(config, key, parser.(value))
  end

  defp parse_float(str) do
    case Float.parse(str) do
      {float_val, ""} -> float_val
      _ -> raise("Invalid float value: #{str}")
    end
  end

  defp parse_integer(str) do
    case Integer.parse(str) do
      {int_val, ""} -> int_val
      _ -> raise("Invalid integer value: #{str}")
    end
  end

  defp add_thinking_budget(config, value) do
    parsed_value =
      case value do
        # Dynamic thinking
        "-1" ->
          -1

        # No thinking
        "0" ->
          0

        str ->
          case Integer.parse(str) do
            {int_val, ""} when int_val > 0 ->
              int_val

            _ ->
              raise(
                "Invalid thinking budget: #{str}. Use -1 for dynamic, 0 to disable, or positive integer"
              )
          end
      end

    Map.put(config, "thinkingConfig", %{"thinkingBudget" => parsed_value})
  end

  defp evaluation_response_schema do
    %{
      "type" => "object",
      "properties" => %{
        "evaluations" => %{
          "type" => "array",
          "items" => %{
            "type" => "object",
            "properties" => %{
              "linkedin_id" => %{"type" => "string"},
              "evaluation" => %{
                "type" => "string",
                "enum" => ["ACCEPT", "REJECT"]
              }
            },
            "required" => ["linkedin_id", "evaluation"]
          }
        }
      },
      "required" => ["evaluations"]
    }
  end

  defp save_batch_results(batch, batch_num, response, output_file) do
    # Parse response for evaluations (response is already validated)
    results =
      response
      |> String.split("\n")
      |> Enum.filter(&String.contains?(&1, ":"))
      |> Enum.map(fn line ->
        case String.split(line, ":", parts: 2) do
          [id, evaluation] ->
            %{
              linkedin_id: String.trim(id),
              evaluation: String.trim(evaluation)
            }

          _ ->
            nil
        end
      end)
      |> Enum.filter(&(&1 != nil))

    # Count accepted/rejected
    accepted = Enum.count(results, &(&1.evaluation == "ACCEPT"))
    rejected = Enum.count(results, &(&1.evaluation == "REJECT"))

    # Save to output file
    batch_result = %{
      batch: batch_num,
      batch_size: length(batch),
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      results: results
    }

    File.write!(output_file, JSON.encode!(batch_result) <> "\n", [:append])

    {accepted, rejected}
  end

  defp log_progress(batch_num, batch_size, progress_file) do
    timestamp = DateTime.utc_now() |> DateTime.to_iso8601()
    progress = "#{timestamp}: Batch #{batch_num}, Profiles processed: #{batch_size}"

    File.write!(progress_file, progress <> "\n", [:append])
    Logger.info("Progress: Batch #{batch_num} complete (#{batch_size} profiles)")
  end

  defp calculate_wait_time(rpm) do
    # Wait time to respect rate limit
    # Add 10% buffer
    round(60_000 / rpm * 1.1)
  end

  defp process_profiles_batch(config) do
    %{
      model: model,
      batch_size: batch_size,
      start_from: start_from,
      stop_at: stop_at,
      prompt: prompt,
      output_file: output_file,
      profiles_to_process: profiles_to_process
    } = config

    IO.puts("Creating batch job for #{profiles_to_process} profiles...")

    # Generate batch requests (returns list of request maps)
    batch_requests = create_batch_requests(model, batch_size, start_from, prompt, stop_at)

    # Create batch job (async processing)
    case create_batch_job(model, batch_requests) do
      {:ok, batch_name} ->
        # Save batch info to output file
        batch_info = %{
          batch_id: batch_name,
          model: model,
          batch_size: batch_size,
          profiles_to_process: profiles_to_process,
          start_from: start_from,
          stop_at: stop_at,
          created_at: DateTime.utc_now() |> DateTime.to_iso8601(),
          status: "submitted"
        }

        # Create batch info file alongside the planned output file
        batch_info_file = String.replace(output_file, ".jsonl", "_batch_info.json")
        File.write!(batch_info_file, JSON.encode!(batch_info))

        IO.puts("Batch job submitted successfully!")
        IO.puts("Batch ID: #{batch_name}")
        IO.puts("Batch info saved to: #{batch_info_file}")
        IO.puts("Batch processing will take ~24 hours with 50% cost savings.")
        IO.puts("")
        IO.puts("Next steps:")
        IO.puts("1. Check status: elixir evaluate_profiles.exs check_batch #{batch_name}")

        IO.puts(
          "2. Retrieve results when complete: elixir evaluate_profiles.exs retrieve_batch #{batch_name}"
        )

      {:error, reason} ->
        IO.puts("Failed to create batch job: #{reason}")
    end
  end

  defp create_batch_requests(_model, batch_size, start_from, prompt, stop_at) do
    # Reuse the streaming logic
    config = %{start_from: start_from, stop_at: stop_at}

    requests =
      stream_profiles(config)
      |> Stream.chunk_every(batch_size)
      |> Stream.with_index(1)
      |> Enum.map(fn {batch, batch_num} ->
        # Format profiles for this batch
        formatted_profiles = format_batch_profiles(batch)

        # Create evaluation prompt
        full_prompt = create_evaluation_prompt(formatted_profiles, prompt)

        # Create GenerateContentRequest for batch with structured output
        %{
          "batch_#{batch_num}" => %{
            "contents" => [
              %{
                "parts" => [
                  %{"text" => full_prompt}
                ]
              }
            ],
            "generationConfig" => build_generation_config()
          }
        }
      end)

    # Return as a map of requests for inline batch
    requests
  end

  defp create_batch_job(model, batch_requests) do
    rate_config = Map.get(@rate_limits, model)

    case rate_config.provider do
      :gemini -> create_gemini_batch_job(model, batch_requests)
      :anthropic -> create_anthropic_batch_job(model, batch_requests)
      :openai -> create_openai_batch_job(model, batch_requests)
    end
  end

  defp create_gemini_batch_job(model, batch_requests) do
    api_key = System.get_env("GEMINI_API_KEY")
    if !api_key, do: raise("GEMINI_API_KEY environment variable not set")

    # Use the correct batch endpoint with model name
    url = "https://generativelanguage.googleapis.com/v1beta/models/#{model}:batchGenerateContent"

    # Convert request maps into proper batch format for the API
    inline_requests =
      batch_requests
      |> Enum.flat_map(fn request_map ->
        Map.to_list(request_map)
      end)
      |> Enum.with_index()
      |> Enum.map(fn {{batch_id, request_content}, index} ->
        %{
          "request" => request_content,
          "metadata" => %{
            "batch_id" => batch_id,
            "index" => index
          }
        }
      end)

    # Create the proper batch API payload structure
    payload = %{
      "batch" => %{
        "display_name" =>
          "Profile Evaluation Batch - #{DateTime.utc_now() |> DateTime.to_iso8601()}",
        "input_config" => %{
          "requests" => %{
            "requests" => inline_requests
          }
        }
      }
    }

    headers = [
      {"x-goog-api-key", api_key},
      {"Content-Type", "application/json"}
    ]

    IO.puts("Creating Gemini batch job with #{length(inline_requests)} requests...")

    # Batch job creation
    case Req.post(url, json: payload, headers: headers, receive_timeout: 60_000) do
      {:ok, %{status: 200, body: body}} ->
        batch_name = body["name"]
        IO.puts("Gemini batch job created: #{batch_name}")
        IO.puts("Monitor with: elixir evaluate_profiles.exs check_batch #{batch_name}")
        {:ok, batch_name}

      {:ok, %{status: status, body: body}} ->
        {:error, "HTTP #{status}: #{inspect(body)}"}

      {:error, reason} ->
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  defp create_anthropic_batch_job(model, batch_requests) do
    api_key = System.get_env("ANTHROPIC_API_KEY")
    if !api_key, do: raise("ANTHROPIC_API_KEY environment variable not set")

    url = "https://api.anthropic.com/v1/messages/batches"

    # Convert request maps to Anthropic batch format
    requests =
      batch_requests
      |> Enum.flat_map(fn request_map ->
        Map.to_list(request_map)
      end)
      |> Enum.with_index()
      |> Enum.map(fn {{batch_id, request_content}, index} ->
        # Get the prompt from the Gemini request content
        prompt_text =
          get_in(request_content, ["contents", Access.at(0), "parts", Access.at(0), "text"])

        # Build base params
        base_params = %{
          "model" => model,
          "max_tokens" => 4096,
          "messages" => [
            %{
              "role" => "user",
              "content" => prompt_text
            },
            %{
              "role" => "assistant",
              "content" => "```json\n{\n  \"evaluations\": [\n    {\"linkedin_id\": \""
            }
          ]
        }
        
        # Apply generation config
        params = add_anthropic_generation_config(base_params)
        
        %{
          "custom_id" => "#{batch_id}_#{index}",
          "params" => params
        }
      end)

    payload = %{
      "requests" => requests
    }

    headers = [
      {"x-api-key", api_key},
      {"anthropic-version", "2023-06-01"},
      {"Content-Type", "application/json"}
    ]

    IO.puts("Creating Anthropic batch job with #{length(requests)} requests...")

    case Req.post(url, json: payload, headers: headers, receive_timeout: 60_000) do
      {:ok, %{status: status, body: body}} when status in [200, 201] ->
        batch_id = body["id"]
        IO.puts("✅ Anthropic batch job created: #{batch_id}")
        
        # Display batch details
        if expires_at = body["expires_at"] do
          # expires_at is already a string from Anthropic API
          IO.puts("⏰ Expires at: #{expires_at}")
        end
        
        if request_counts = body["request_counts"] do
          total = Map.values(request_counts) |> Enum.sum()
          IO.puts("📊 Requests: #{total} total")
        end
        
        IO.puts("Monitor with: elixir evaluate_profiles.exs check_batch #{batch_id}")
        {:ok, batch_id}

      {:ok, %{status: status, body: body}} ->
        {:error, "HTTP #{status}: #{inspect(body)}"}

      {:error, reason} ->
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  defp create_openai_batch_job(model, batch_requests) do
    api_key = System.get_env("OPENAI_API_KEY")
    if !api_key, do: raise("OPENAI_API_KEY environment variable not set")

    # Step 1: Create a JSONL file with batch requests
    timestamp = DateTime.utc_now() |> DateTime.to_unix()
    input_file_name = "batch_input_#{model}_#{timestamp}.jsonl"
    input_file_path = Path.join(@output_dir, input_file_name)

    # Convert request maps to OpenAI batch format
    requests =
      batch_requests
      |> Enum.flat_map(fn request_map ->
        Map.to_list(request_map)
      end)
      |> Enum.with_index()
      |> Enum.map(fn {{batch_id, request_content}, index} ->
        # Get the prompt from the Gemini request content
        prompt_text =
          get_in(request_content, ["contents", Access.at(0), "parts", Access.at(0), "text"])

        # Build base payload
        base_payload = %{
          "model" => model,
          "messages" => [
            %{
              "role" => "user",
              "content" => prompt_text
            }
          ],
          "max_completion_tokens" => 4096
        }

        # Apply generation config (handles temperature/top_p restrictions)
        payload = add_openai_generation_config(base_payload)

        %{
          "custom_id" => "#{batch_id}_#{index}",
          "method" => "POST",
          "url" => "/v1/chat/completions",
          "body" => payload
        }
      end)

    # Write JSONL file
    jsonl_content =
      requests
      |> Enum.map(&JSON.encode!/1)
      |> Enum.join("\n")

    File.write!(input_file_path, jsonl_content)
    IO.puts("Created OpenAI batch input file: #{input_file_path}")

    # Step 2: Upload file to OpenAI
    case upload_file_to_openai(input_file_path, api_key) do
      {:ok, file_id} ->
        IO.puts("Uploaded file to OpenAI: #{file_id}")

        # Step 3: Create batch job
        url = "https://api.openai.com/v1/batches"

        payload = %{
          "input_file_id" => file_id,
          "endpoint" => "/v1/chat/completions",
          "completion_window" => "24h",
          "metadata" => %{
            "model" => model,
            "created_at" => DateTime.utc_now() |> DateTime.to_iso8601()
          }
        }

        headers = [
          {"Authorization", "Bearer #{api_key}"},
          {"Content-Type", "application/json"}
        ]

        IO.puts("Creating OpenAI batch job with #{length(requests)} requests...")

        case Req.post(url, json: payload, headers: headers, receive_timeout: 60_000) do
          {:ok, %{status: status, body: body}} when status in [200, 201] ->
            batch_id = body["id"]
            IO.puts("✅ OpenAI batch job created: #{batch_id}")
            
            # Display batch details
            if expires_at = body["expires_at"] do
              expires_time = DateTime.from_unix!(expires_at) |> DateTime.to_string()
              IO.puts("⏰ Expires at: #{expires_time}")
            end
            
            if request_counts = body["request_counts"] do
              IO.puts("📊 Requests: #{request_counts["total"]} total")
            end
            
            IO.puts("Monitor with: elixir evaluate_profiles.exs check_batch #{batch_id}")
            {:ok, batch_id}

          {:ok, %{status: status, body: body}} ->
            {:error, "HTTP #{status}: #{inspect(body)}"}

          {:error, reason} ->
            {:error, "Request failed: #{inspect(reason)}"}
        end

      {:error, reason} ->
        {:error, "Failed to upload file: #{reason}"}
    end
  end

  defp upload_file_to_openai(file_path, api_key) do
    url = "https://api.openai.com/v1/files"

    # Read the file content
    file_content = File.read!(file_path)
    file_name = Path.basename(file_path)

    # Create multipart form data
    form_fields = [
      purpose: "batch",
      file: {file_content, filename: file_name}
    ]

    headers = [
      {"Authorization", "Bearer #{api_key}"}
    ]

    case Req.post(url, form_multipart: form_fields, headers: headers, receive_timeout: 60_000) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, body["id"]}

      {:ok, %{status: status, body: body}} ->
        {:error, "HTTP #{status}: #{inspect(body)}"}

      {:error, reason} ->
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  defp run_tsv_evaluation(model, batch_size, tsv_file) do
    # Track start time
    start_time = System.monotonic_time(:millisecond)

    # Initialize counters
    Process.put(:total_input_tokens, 0)
    Process.put(:total_output_tokens, 0)
    Process.put(:total_profiles_processed, 0)

    # Verify TSV file exists
    unless File.exists?(tsv_file) do
      IO.puts("❌ TSV file not found: #{tsv_file}")
      exit({:shutdown, 1})
    end

    IO.puts("📊 Processing TSV file: #{tsv_file}")
    IO.puts("🤖 Model: #{model}")
    IO.puts("📦 Batch size: #{(batch_size > 0 && batch_size) || "auto-optimal"}")
    IO.puts("⚡ Mode: interactive (TSV mode only supports interactive processing)")

    # Ensure output directory exists
    File.mkdir_p!(@output_dir)

    # Load TSV data
    {headers, rows} = load_tsv_data(tsv_file)

    # Verify data exists
    if Enum.empty?(headers) or Enum.empty?(rows) do
      IO.puts("❌ TSV file is empty or invalid")
      exit({:shutdown, 1})
    end

    IO.puts("📋 Loaded #{length(rows)} rows with #{length(headers)} columns")
    IO.puts("📝 Profile summary column: #{Enum.at(headers, 0)}")

    # Create output file name
    output_file =
      case Process.get(:output_file) do
        nil ->
          timestamp = DateTime.utc_now() |> DateTime.to_unix()
          base_name = Path.basename(tsv_file, ".tsv")

          Path.join(
            @output_dir,
            "#{base_name}_evaluated_#{String.replace(model, ".", "_")}_#{timestamp}.tsv"
          )

        custom_output ->
          # Use custom output file path
          custom_output
      end

    # Load evaluation prompt
    prompt = load_prompt()

    # Get rate configuration for the model
    rate_config = Map.get(@rate_limits, model)

    unless rate_config do
      IO.puts("❌ Unsupported model: #{model}")
      exit({:shutdown, 1})
    end

    # Create optimal batch size if not specified
    optimal_batch_size =
      if batch_size > 0, do: batch_size, else: rate_config.optimal_batch

    # Create config for batch processing
    config = %{
      model: model,
      prompt: prompt,
      output_file: output_file,
      progress_file: nil,
      rate_config: rate_config,
      tsv_headers: headers,
      tsv_mode: true
    }

    # Log generation parameters at verbosity level 1+
    verbose = Process.get(:verbose, 0)
    if verbose >= 1 do
      IO.puts("\n⚙️  Generation parameters:")
      
      # Check if model supports temperature
      model_lower = String.downcase(model)
      temperature_supported = not (String.contains?(model_lower, "gpt-5") or 
                                   String.contains?(model_lower, "o1") or
                                   String.starts_with?(model_lower, "o1-"))
      
      if temperature_supported do
        # Temperature (only if explicitly set)
        temp = Process.get(:temperature)
        if temp do
          IO.puts("   • Temperature: #{temp}")
        else
          IO.puts("   • Temperature: not set (using model default)")
        end
        
        # Top-P (only if explicitly set)
        top_p = Process.get(:top_p)
        if top_p do
          IO.puts("   • Top-P: #{top_p}")
        else
          IO.puts("   • Top-P: not set (using model default)")
        end
      else
        IO.puts("   • Temperature: not supported by #{model}")
        IO.puts("   • Top-P: not supported by #{model}")
      end
      
      # Top-K (Gemini only)
      if String.starts_with?(model, "gemini") do
        top_k = Process.get(:top_k)
        IO.puts("   • Top-K: #{top_k || "not set"}")
      end
      
      # Thinking budget (Gemini thinking models only)
      if String.contains?(model, "thinking") do
        thinking_budget = Process.get(:thinking_budget)
        IO.puts("   • Thinking budget: #{thinking_budget || "32768 (default)"}")
      end
    end
    
    IO.puts("\n🚀 Starting TSV evaluation with batch size: #{optimal_batch_size}")

    # Process in batches, reusing existing batch processing logic
    total_processed =
      rows
      |> Enum.with_index()
      |> Enum.chunk_every(optimal_batch_size)
      |> Enum.with_index(1)
      |> Enum.reduce(0, fn {batch, batch_num}, acc ->
        IO.puts(
          "Processing batch #{batch_num}/#{div(length(rows) - 1, optimal_batch_size) + 1} (#{length(batch)} profiles)..."
        )

        # Convert TSV rows to format expected by process_batch
        # Each item becomes {profile_summary_with_line_number, original_row_index}
        tsv_batch =
          batch
          |> Enum.map(fn {row, row_index} ->
            profile_summary = Enum.at(row, 0) || ""
            # Store both the profile summary + line number, and the full row for output
            {{profile_summary, row_index + 1}, {row, row_index}}
          end)

        # Process this batch using existing logic
        process_tsv_batch(tsv_batch, batch_num, config)

        acc + length(batch)
      end)

    IO.puts("✅ TSV evaluation completed!")
    IO.puts("📄 Output saved to: #{output_file}")
    IO.puts("📊 Processed #{total_processed} rows")

    # Report token usage and runtime with profile count
    report_final_stats(start_time, model, total_processed)
  end

  defp load_tsv_data(tsv_file) do
    lines =
      File.read!(tsv_file)
      |> String.split("\n")
      |> Enum.map(&String.trim/1)
      |> Enum.filter(&(&1 != ""))

    case lines do
      [header_line | data_lines] ->
        headers = String.split(header_line, "\t")
        rows = Enum.map(data_lines, &String.split(&1, "\t"))
        {headers, rows}

      [] ->
        {[], []}
    end
  end

  defp process_tsv_batch(batch, batch_num, config, retry_count \\ 0) do
    %{
      model: model,
      prompt: prompt,
      output_file: output_file,
      rate_config: rate_config,
      tsv_headers: headers,
      tsv_mode: true
    } = config

    max_retries = 3

    Logger.info(
      "Processing TSV batch #{batch_num}#{if retry_count > 0, do: " (retry #{retry_count}/#{max_retries})", else: ""}..."
    )

    # Format profiles for evaluation using our new TSV format
    formatted_profiles = format_tsv_batch_profiles(batch)

    # Create evaluation prompt
    full_prompt = create_evaluation_prompt(formatted_profiles, prompt)

    # Show full prompt at verbosity 4+
    verbose = Process.get(:verbose, 0)

    if verbose >= 4 do
      IO.puts("\n📝 Full prompt for batch #{batch_num}:")
      IO.puts(full_prompt)
      IO.puts("")
    end

    # Wait for rate limit if needed
    bucket_name = "api_#{model}"
    time_scale = 60_000

    case ExRated.check_rate(bucket_name, time_scale, rate_config.rpm) do
      {:ok, _count} ->
        # Make API call using existing function
        case call_api(model, full_prompt) do
          {:ok, %{text: response, usage: usage}} ->
            # Store usage for reporting
            update_token_usage(usage, model)

            verbose = Process.get(:verbose, 0)

            # Check if response is empty or nil
            if response == nil or response == "" do
              if retry_count < max_retries do
                if verbose >= 1 do
                  IO.puts("⚠️  Empty response for batch #{batch_num}, retrying...")
                end

                Process.sleep(2000)
                process_tsv_batch(batch, batch_num, config, retry_count + 1)
              else
                IO.puts("❌ Empty response after #{max_retries} retries for batch #{batch_num}")
                # Only as last resort, save as ERROR
                error_evaluations = Enum.map(batch, fn _ -> "ERROR" end)

                save_tsv_batch_results_with_evaluations(
                  batch,
                  error_evaluations,
                  headers,
                  model,
                  output_file,
                  batch_num == 1
                )
              end
            else
              # Show full response at verbosity 3+
              if verbose >= 3 do
                IO.puts("\n📄 Full response for batch #{batch_num}:")
                IO.puts(response)
                IO.puts("")
              end

              # Parse evaluations to check if we got valid results
              parsed_evaluations = parse_evaluation_response(response, verbose)
              evaluations = Enum.map(parsed_evaluations, fn eval -> eval["evaluation"] end)

              # Check if we got enough valid evaluations
              if length(evaluations) < length(batch) do
                if retry_count < max_retries do
                  if verbose >= 1 do
                    IO.puts(
                      "⚠️  Only got #{length(evaluations)}/#{length(batch)} evaluations for batch #{batch_num}, retrying..."
                    )
                  end

                  Process.sleep(2000)
                  process_tsv_batch(batch, batch_num, config, retry_count + 1)
                else
                  IO.puts(
                    "⚠️  Incomplete evaluations after #{max_retries} retries, using what we got"
                  )

                  save_tsv_batch_results(
                    batch,
                    response,
                    headers,
                    model,
                    output_file,
                    batch_num == 1
                  )
                end
              else
                # Success - save the results
                save_tsv_batch_results(
                  batch,
                  response,
                  headers,
                  model,
                  output_file,
                  batch_num == 1
                )
              end
            end

          {:error, reason} ->
            if retry_count < max_retries do
              if verbose >= 1 do
                IO.puts("⚠️  API call failed for batch #{batch_num}: #{reason}. Retrying...")
              end

              # Exponential backoff
              Process.sleep(2000 * (retry_count + 1))
              process_tsv_batch(batch, batch_num, config, retry_count + 1)
            else
              IO.puts(
                "❌ API call failed after #{max_retries} retries for batch #{batch_num}: #{reason}"
              )

              # Only as last resort, save as ERROR
              error_evaluations = Enum.map(batch, fn _ -> "ERROR" end)

              save_tsv_batch_results_with_evaluations(
                batch,
                error_evaluations,
                headers,
                model,
                output_file,
                batch_num == 1
              )
            end
        end

      {:error, _limit} ->
        # Calculate proper wait time: we need to wait for at least one slot to become available
        # For rate limiting, wait slightly longer than the minimum interval between requests
        wait_time = calculate_wait_time(rate_config.rpm)
        IO.puts("⏳ Rate limit reached, waiting #{wait_time}ms...")
        :timer.sleep(wait_time)
        # Don't increment retry count for rate limits
        process_tsv_batch(batch, batch_num, config, retry_count)
    end
  end

  defp format_tsv_batch_profiles(batch) do
    batch
    |> Enum.map(fn {profile_data, _row_data} -> format_profile(profile_data) end)
    |> Enum.join("\n\n")
  end

  defp save_tsv_batch_results(batch, response, headers, model, output_file, is_first_batch) do
    # Parse evaluations from response using existing function
    verbose = Process.get(:verbose, 0)
    parsed_evaluations = parse_evaluation_response(response, verbose)

    # Extract just the evaluation values for TSV output
    evaluations = Enum.map(parsed_evaluations, fn eval -> eval["evaluation"] end)

    # Ensure we have the right number of evaluations
    batch_size = length(batch)
    eval_count = length(evaluations)

    final_evaluations =
      if eval_count != batch_size do
        # We should have retried already, but if we still have a mismatch, handle it
        if eval_count < batch_size do
          IO.puts(
            "⚠️  Using #{eval_count} evaluations, padding #{batch_size - eval_count} with 'RETRY_FAILED'"
          )

          evaluations ++ List.duplicate("RETRY_FAILED", batch_size - eval_count)
        else
          # Too many evaluations, truncate
          Enum.take(evaluations, batch_size)
        end
      else
        evaluations
      end

    save_tsv_batch_results_with_evaluations(
      batch,
      final_evaluations,
      headers,
      model,
      output_file,
      is_first_batch
    )
  end

  defp save_tsv_batch_results_with_evaluations(
         batch,
         evaluations,
         headers,
         model,
         output_file,
         is_first_batch
       ) do
    # Create output headers (add model column with temperature if explicitly set)
    temp = Process.get(:temperature)
    model_lower = String.downcase(model)
    
    # Check if temperature was actually set and is supported by the model
    temperature_restricted = String.contains?(model_lower, "gpt-5") or 
                           String.contains?(model_lower, "o1") or
                           String.starts_with?(model_lower, "o1-")
    
    model_header = 
      if temp && !temperature_restricted do
        # Temperature was explicitly set and is supported
        "#{model} t=#{temp}"
      else
        # No temperature set or not supported
        model
      end
    output_headers = headers ++ [model_header]

    # Write headers if this is the first batch
    if is_first_batch do
      header_line = Enum.join(output_headers, "\t") <> "\n"
      File.write!(output_file, header_line)
    end

    # Write data rows with evaluations
    output_lines =
      batch
      |> Enum.zip(evaluations)
      |> Enum.map(fn {{_profile_data, {row, _row_index}}, evaluation} ->
        output_row = row ++ [evaluation]
        Enum.join(output_row, "\t") <> "\n"
      end)
      # Convert list of strings to single string
      |> Enum.join("")

    File.write!(output_file, output_lines, [:append])
  end

  defp show_summary(output_file) do
    Logger.info("")
    Logger.info("=== Evaluation Complete ===")

    # Count results
    {total, accepted, rejected} =
      output_file
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.filter(&(&1 != ""))
      |> Stream.map(fn line ->
        case JSON.decode(line) do
          {:ok, %{"results" => results}} -> results
          _ -> []
        end
      end)
      |> Enum.reduce({0, 0, 0}, fn results, {total_acc, acc_acc, rej_acc} ->
        new_total = total_acc + length(results)

        new_accepted =
          acc_acc + Enum.count(results, &String.contains?(&1["evaluation"], "ACCEPT"))

        new_rejected =
          rej_acc + Enum.count(results, &String.contains?(&1["evaluation"], "REJECT"))

        {new_total, new_accepted, new_rejected}
      end)

    Logger.info("Total profiles evaluated: #{total}")
    Logger.info("Accepted: #{accepted}")
    Logger.info("Rejected: #{rejected}")
    Logger.info("Results saved to: #{output_file}")
  end

  defp check_batch_status(batch_id) do
    # Determine provider from batch_id format
    cond do
      String.starts_with?(batch_id, "batches/") ->
        check_gemini_batch_status(batch_id)

      String.starts_with?(batch_id, "batch_") ->
        check_openai_batch_status(batch_id)

      String.starts_with?(batch_id, "msgbatch_") ->
        check_anthropic_batch_status(batch_id)

      true ->
        # Try to detect from content if unclear
        check_anthropic_batch_status(batch_id)
    end
  end

  defp check_gemini_batch_status(batch_id) do
    api_key = System.get_env("GEMINI_API_KEY")
    if !api_key, do: raise("GEMINI_API_KEY environment variable not set")

    # Ensure batch_id has the correct format (should start with "batches/")
    batch_name =
      if String.starts_with?(batch_id, "batches/") do
        batch_id
      else
        "batches/#{batch_id}"
      end

    url = "https://generativelanguage.googleapis.com/v1beta/#{batch_name}"
    headers = [{"x-goog-api-key", api_key}]

    case Req.get(url, headers: headers, receive_timeout: 30_000) do
      {:ok, %{status: 200, body: body}} ->
        # Check both possible locations for state
        state = body["metadata"]["state"] || body["state"]
        done = body["done"]

        IO.puts("Batch Status: #{state}")
        IO.puts("Done: #{done}")

        case state do
          "BATCH_STATE_SUCCEEDED" ->
            IO.puts("Batch completed successfully!")

            IO.puts(
              "Retrieve results with: elixir evaluate_profiles.exs retrieve_batch #{batch_id}"
            )

          "BATCH_STATE_FAILED" ->
            IO.puts("Batch failed: #{inspect(body["error"])}")

          "BATCH_STATE_IN_PROGRESS" ->
            IO.puts("Batch is still processing...")

          "BATCH_STATE_RUNNING" ->
            IO.puts("Batch is running...")

          "BATCH_STATE_PENDING" ->
            IO.puts("Batch is queued and waiting to start...")

          _ ->
            IO.puts("Unknown state: #{inspect(state)}")
        end

        # Show batch statistics if available
        if batch_stats = get_in(body, ["metadata", "batchStats"]) do
          IO.puts("Batch Statistics:")
          IO.puts("  Total requests: #{batch_stats["requestCount"]}")
          IO.puts("  Successful: #{batch_stats["successfulRequestCount"]}")

          if failed = batch_stats["failedRequestCount"] do
            IO.puts("  Failed: #{failed}")
          end
        end

        # Show timing information
        if metadata = body["metadata"] do
          if create_time = metadata["createTime"] do
            IO.puts("Created: #{create_time}")
          end

          if end_time = metadata["endTime"] do
            IO.puts("Completed: #{end_time}")
          end
        end

      {:ok, %{status: status, body: body}} ->
        IO.puts("Error checking batch status: HTTP #{status}: #{inspect(body)}")

      {:error, reason} ->
        IO.puts("Failed to check batch status: #{inspect(reason)}")
    end
  end

  defp check_anthropic_batch_status(batch_id) do
    api_key = System.get_env("ANTHROPIC_API_KEY")
    if !api_key, do: raise("ANTHROPIC_API_KEY environment variable not set")

    url = "https://api.anthropic.com/v1/messages/batches/#{batch_id}"

    headers = [
      {"x-api-key", api_key},
      {"anthropic-version", "2023-06-01"}
    ]

    case Req.get(url, headers: headers, receive_timeout: 30_000) do
      {:ok, %{status: 200, body: body}} ->
        processing_status = body["processing_status"]
        
        # Display status with emoji
        status_emoji = case processing_status do
          "in_progress" -> "⚙️"
          "canceling" -> "🚫"
          "ended" -> "✅"
          _ -> "❓"
        end

        IO.puts("\n#{status_emoji} Batch Status: #{processing_status}")
        
        # Display request counts if available
        if request_counts = body["request_counts"] do
          processing = request_counts["processing"] || 0
          succeeded = request_counts["succeeded"] || 0
          errored = request_counts["errored"] || 0
          canceled = request_counts["canceled"] || 0
          expired = request_counts["expired"] || 0
          
          total = processing + succeeded + errored + canceled + expired
          
          if total > 0 do
            completed_pct = if total > 0, do: Float.round((succeeded + errored + canceled + expired) / total * 100, 1), else: 0
            IO.puts("📊 Progress: #{succeeded} succeeded, #{errored} errored, #{processing} processing (#{completed_pct}% complete)")
          end
        end

        case processing_status do
          "in_progress" ->
            IO.puts("Batch is processing...")

          "canceling" ->
            IO.puts("Batch is being cancelled...")

          "ended" ->
            IO.puts("Batch processing ended!")
            
            if ended_at = body["ended_at"] do
              IO.puts("Ended at: #{ended_at}")
            end

            IO.puts("Retrieve results with: elixir evaluate_profiles.exs retrieve_batch #{batch_id}")

          _ ->
            IO.puts("Unknown status: #{inspect(processing_status)}")
        end


        # Show timing information
        if created_at = body["created_at"] do
          IO.puts("Created: #{created_at}")
        end

        if ended_at = body["ended_at"] do
          IO.puts("Completed: #{ended_at}")
        end

      {:ok, %{status: status, body: body}} ->
        IO.puts("Error checking Anthropic batch status: HTTP #{status}: #{inspect(body)}")

      {:error, reason} ->
        IO.puts("Failed to check Anthropic batch status: #{inspect(reason)}")
    end
  end

  defp check_openai_batch_status(batch_id) do
    api_key = System.get_env("OPENAI_API_KEY")
    if !api_key, do: raise("OPENAI_API_KEY environment variable not set")

    url = "https://api.openai.com/v1/batches/#{batch_id}"

    headers = [
      {"Authorization", "Bearer #{api_key}"}
    ]

    case Req.get(url, headers: headers, receive_timeout: 30_000) do
      {:ok, %{status: 200, body: body}} ->
        status = body["status"]
        
        # Display status with emoji
        status_emoji = case status do
          "validating" -> "🔍"
          "in_progress" -> "⚙️"
          "finalizing" -> "📦"
          "completed" -> "✅"
          "failed" -> "❌"
          "expired" -> "⏰"
          "cancelling" -> "🚫"
          "cancelled" -> "⛔"
          _ -> "❓"
        end

        IO.puts("\n#{status_emoji} Batch Status: #{status}")
        
        # Display request counts if available
        if request_counts = body["request_counts"] do
          total = request_counts["total"] || 0
          completed = request_counts["completed"] || 0
          failed = request_counts["failed"] || 0
          
          if total > 0 do
            percentage = Float.round(completed / total * 100, 1)
            IO.puts("📊 Progress: #{completed}/#{total} completed (#{percentage}%), #{failed} failed")
          end
        end

        case status do
          "validating" ->
            IO.puts("Batch is being validated...")

          "in_progress" ->
            IO.puts("Batch is processing...")
            
            if in_progress_at = body["in_progress_at"] do
              start_time = DateTime.from_unix!(in_progress_at) |> DateTime.to_string()
              IO.puts("Started at: #{start_time}")
            end

          "finalizing" ->
            IO.puts("Batch is finalizing...")

          "completed" ->
            IO.puts("Batch completed successfully!")
            
            if completed_at = body["completed_at"] do
              complete_time = DateTime.from_unix!(completed_at) |> DateTime.to_string()
              IO.puts("Completed at: #{complete_time}")
            end

            IO.puts(
              "Retrieve results with: elixir evaluate_profiles.exs retrieve_batch #{batch_id}"
            )

          "failed" ->
            IO.puts("Batch failed")

            if errors = body["errors"] do
              IO.puts("Errors: #{inspect(errors)}")
            end
            
            if failed_at = body["failed_at"] do
              fail_time = DateTime.from_unix!(failed_at) |> DateTime.to_string()
              IO.puts("Failed at: #{fail_time}")
            end

          "expired" ->
            IO.puts("Batch expired - results are no longer available")
            
            if expired_at = body["expired_at"] do
              expire_time = DateTime.from_unix!(expired_at) |> DateTime.to_string()
              IO.puts("Expired at: #{expire_time}")
            end

          "cancelling" ->
            IO.puts("Batch is being cancelled...")

          "cancelled" ->
            IO.puts("Batch was cancelled")
            
            if cancelled_at = body["cancelled_at"] do
              cancel_time = DateTime.from_unix!(cancelled_at) |> DateTime.to_string()
              IO.puts("Cancelled at: #{cancel_time}")
            end

          _ ->
            IO.puts("Unknown status: #{inspect(status)}")
        end

        # Show additional timing information
        if created_at = body["created_at"] do
          created_time = DateTime.from_unix!(created_at) |> DateTime.to_iso8601()
          IO.puts("Created: #{created_time}")
        end

        if completed_at = body["completed_at"] do
          completed_time = DateTime.from_unix!(completed_at) |> DateTime.to_iso8601()
          IO.puts("Completed: #{completed_time}")
        end

      {:ok, %{status: status, body: body}} ->
        IO.puts("Error checking OpenAI batch status: HTTP #{status}: #{inspect(body)}")

      {:error, reason} ->
        IO.puts("Failed to check OpenAI batch status: #{inspect(reason)}")
    end
  end

  defp retrieve_batch_results(batch_id) do
    # Determine provider from batch_id format
    cond do
      String.starts_with?(batch_id, "batches/") ->
        retrieve_gemini_batch_results(batch_id)

      String.starts_with?(batch_id, "batch_") ->
        retrieve_openai_batch_results(batch_id)

      String.starts_with?(batch_id, "msgbatch_") ->
        retrieve_anthropic_batch_results(batch_id)

      true ->
        # Try to detect from content if unclear
        retrieve_anthropic_batch_results(batch_id)
    end
  end

  defp retrieve_gemini_batch_results(batch_id) do
    api_key = System.get_env("GEMINI_API_KEY")
    if !api_key, do: raise("GEMINI_API_KEY environment variable not set")

    # Ensure batch_id has the correct format (should start with "batches/")
    batch_name =
      if String.starts_with?(batch_id, "batches/") do
        batch_id
      else
        "batches/#{batch_id}"
      end

    url = "https://generativelanguage.googleapis.com/v1beta/#{batch_name}"
    headers = [{"x-goog-api-key", api_key}]

    case Req.get(url, headers: headers, receive_timeout: 30_000) do
      {:ok, %{status: 200, body: body}} ->
        state = body["metadata"]["state"] || body["state"]
        done = body["done"]

        if state != "BATCH_STATE_SUCCEEDED" or not done do
          IO.puts("Batch not ready yet. Current state: #{state}, Done: #{done}")

          IO.puts(
            "Use 'elixir evaluate_profiles.exs check_batch #{batch_id}' to monitor progress."
          )
        else
          # Check for inline responses in the correct location
          if inline_responses = get_in(body, ["response", "inlinedResponses", "inlinedResponses"]) do
            IO.puts("Processing inline batch results...")
            process_inline_batch_results(inline_responses, batch_id)
          else
            IO.puts("No inline responses found in batch response")
            IO.puts("Full response structure: #{inspect(body)}")
          end
        end

      {:ok, %{status: status, body: body}} ->
        IO.puts("Error retrieving batch: HTTP #{status}: #{inspect(body)}")

      {:error, reason} ->
        IO.puts("Failed to retrieve batch: #{inspect(reason)}")
    end
  end

  defp retrieve_anthropic_batch_results(batch_id) do
    api_key = System.get_env("ANTHROPIC_API_KEY")
    if !api_key, do: raise("ANTHROPIC_API_KEY environment variable not set")

    # First check batch status
    url = "https://api.anthropic.com/v1/messages/batches/#{batch_id}"

    headers = [
      {"x-api-key", api_key},
      {"anthropic-version", "2023-06-01"}
    ]

    case Req.get(url, headers: headers, receive_timeout: 30_000) do
      {:ok, %{status: 200, body: body}} ->
        processing_status = body["processing_status"]

        if processing_status != "ended" do
          IO.puts("Batch not ready yet. Current status: #{processing_status}")

          IO.puts(
            "Use 'elixir evaluate_profiles.exs check_batch #{batch_id}' to monitor progress."
          )
        else
          # Display final statistics
          if request_counts = body["request_counts"] do
            succeeded = request_counts["succeeded"] || 0
            errored = request_counts["errored"] || 0
            canceled = request_counts["canceled"] || 0
            expired = request_counts["expired"] || 0
            
            IO.puts("\n📊 Final Statistics:")
            IO.puts("   • Succeeded: #{succeeded}")
            IO.puts("   • Errored: #{errored}")
            IO.puts("   • Canceled: #{canceled}")
            IO.puts("   • Expired: #{expired}")
          end
          
          # Get results URL
          if results_url = body["results_url"] do
            IO.puts("\n📥 Downloading batch results...")
            download_anthropic_batch_results(results_url, batch_id, headers)
          else
            IO.puts("No results URL found in ended batch")
            IO.puts("Full response: #{inspect(body)}")
          end
        end

      {:ok, %{status: status, body: body}} ->
        IO.puts("Error retrieving Anthropic batch: HTTP #{status}: #{inspect(body)}")

      {:error, reason} ->
        IO.puts("Failed to retrieve Anthropic batch: #{inspect(reason)}")
    end
  end

  defp download_anthropic_batch_results(results_url, batch_id, headers) do
    case Req.get(results_url, headers: headers, receive_timeout: 60_000) do
      {:ok, %{status: 200, body: body}} ->
        output_file = "data/out/batch_results_#{batch_id}.jsonl"

        # Parse JSONL results
        results =
          String.split(body, "\n")
          |> Enum.filter(&(String.trim(&1) != ""))
          |> Enum.map(fn line ->
            case JSON.decode(line) do
              {:ok, json} -> json
              {:error, _} -> nil
            end
          end)
          |> Enum.filter(&(&1 != nil))

        IO.puts("Processing #{length(results)} Anthropic batch results...")
        
        # Count successful vs failed results
        _succeeded_count = Enum.count(results, fn r -> 
          get_in(r, ["result", "type"]) == "succeeded"
        end)
        
        errored_count = Enum.count(results, fn r -> 
          get_in(r, ["result", "type"]) == "errored"
        end)
        
        if errored_count > 0 do
          IO.puts("⚠️  #{errored_count} requests had errors")
        end

        # Convert Anthropic results to our format
        results
        |> Enum.with_index()
        |> Enum.each(fn {result, index} ->
          if rem(index + 1, 10) == 0 or index + 1 == length(results) do
            IO.puts("Processing result #{index + 1}/#{length(results)}...")
          end
          
          # Check result type
          result_type = get_in(result, ["result", "type"])
          custom_id = result["custom_id"] || "unknown"
          
          case result_type do
            "succeeded" ->
              # Extract text from successful Anthropic response
              text =
                case get_in(result, ["result", "message", "content"]) do
                  [%{"text" => text} | _] ->
                    # Prepend the prefilled content to complete the JSON
                    "```json\n{\n  \"evaluations\": [\n    {\"linkedin_id\": \"" <> text

                  _ ->
                    ""
                end

              if text != "" do
                evaluations = parse_evaluation_response(text)

                formatted_result = %{
                  batch_id: custom_id,
                  index: index,
                  timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
                  results: evaluations
                }

                File.write!(output_file, JSON.encode!(formatted_result) <> "\n", [:append])
              else
                IO.puts("⚠️  No text found in successful result #{custom_id}")
              end
            
            "errored" ->
              error_msg = get_in(result, ["result", "error", "message"]) || "Unknown error"
              IO.puts("❌ Request #{custom_id} failed: #{error_msg}")
              
              # Still save the error result
              formatted_result = %{
                batch_id: custom_id,
                index: index,
                timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
                error: error_msg
              }

              File.write!(output_file, JSON.encode!(formatted_result) <> "\n", [:append])
            
            "canceled" ->
              IO.puts("⛔ Request #{custom_id} was canceled")
            
            _ ->
              IO.puts("❓ Unknown result type '#{result_type}' for request #{custom_id}")
              IO.puts("Result structure: #{inspect(result)}")
          end
        end)

        IO.puts("Results saved to: #{output_file}")

      {:ok, %{status: status, body: body}} ->
        IO.puts("Error downloading results: HTTP #{status}: #{inspect(body)}")

      {:error, reason} ->
        IO.puts("Failed to download results: #{inspect(reason)}")
    end
  end

  defp retrieve_openai_batch_results(batch_id) do
    api_key = System.get_env("OPENAI_API_KEY")
    if !api_key, do: raise("OPENAI_API_KEY environment variable not set")

    # First check batch status
    url = "https://api.openai.com/v1/batches/#{batch_id}"

    headers = [
      {"Authorization", "Bearer #{api_key}"}
    ]

    case Req.get(url, headers: headers, receive_timeout: 30_000) do
      {:ok, %{status: 200, body: body}} ->
        status = body["status"]

        if status != "completed" do
          IO.puts("Batch not ready yet. Current status: #{status}")

          IO.puts(
            "Use 'elixir evaluate_profiles.exs check_batch #{batch_id}' to monitor progress."
          )
        else
          # Display final statistics
          if request_counts = body["request_counts"] do
            total = request_counts["total"] || 0
            completed = request_counts["completed"] || 0
            failed = request_counts["failed"] || 0
            
            IO.puts("\n📊 Final Statistics:")
            IO.puts("   • Total requests: #{total}")
            IO.puts("   • Completed: #{completed}")
            IO.puts("   • Failed: #{failed}")
          end
          
          # Get output file ID
          if output_file_id = body["output_file_id"] do
            IO.puts("\n📥 Downloading batch results...")
            download_openai_batch_results(output_file_id, batch_id, api_key)
          else
            IO.puts("No output file ID found in completed batch")
            IO.puts("Full response: #{inspect(body)}")
          end
          
          # Also download error file if present
          if error_file_id = body["error_file_id"] do
            IO.puts("\n⚠️  Downloading error file...")
            download_openai_error_file(error_file_id, batch_id, api_key)
          end
        end

      {:ok, %{status: status, body: body}} ->
        IO.puts("Error retrieving OpenAI batch: HTTP #{status}: #{inspect(body)}")

      {:error, reason} ->
        IO.puts("Failed to retrieve OpenAI batch: #{inspect(reason)}")
    end
  end

  defp download_openai_error_file(error_file_id, batch_id, api_key) do
    # Get error file content from OpenAI
    url = "https://api.openai.com/v1/files/#{error_file_id}/content"
    
    headers = [
      {"Authorization", "Bearer #{api_key}"}
    ]
    
    case Req.get(url, headers: headers, receive_timeout: 60_000) do
      {:ok, %{status: 200, body: body}} ->
        error_file = "data/out/batch_errors_#{batch_id}.jsonl"
        
        # Parse JSONL error results
        errors =
          String.split(body, "\n")
          |> Enum.filter(&(String.trim(&1) != ""))
          |> Enum.map(fn line ->
            case JSON.decode(line) do
              {:ok, json} -> json
              {:error, _} -> nil
            end
          end)
          |> Enum.filter(&(&1 != nil))
        
        IO.puts("Found #{length(errors)} failed requests")
        
        # Save error file
        File.write!(error_file, body)
        IO.puts("Error details saved to: #{error_file}")
        
        # Display sample errors
        if length(errors) > 0 do
          IO.puts("\nSample error reasons:")
          errors
          |> Enum.take(3)
          |> Enum.each(fn error ->
            custom_id = error["custom_id"] || "unknown"
            error_msg = get_in(error, ["error", "message"]) || "Unknown error"
            IO.puts("   • Request #{custom_id}: #{error_msg}")
          end)
          
          if length(errors) > 3 do
            IO.puts("   • ... and #{length(errors) - 3} more errors")
          end
        end
        
      {:ok, %{status: status, body: body}} ->
        IO.puts("Error downloading error file: HTTP #{status}: #{inspect(body)}")
        
      {:error, reason} ->
        IO.puts("Failed to download error file: #{inspect(reason)}")
    end
  end

  defp download_openai_batch_results(output_file_id, batch_id, api_key) do
    # Get file content from OpenAI
    url = "https://api.openai.com/v1/files/#{output_file_id}/content"

    headers = [
      {"Authorization", "Bearer #{api_key}"}
    ]

    case Req.get(url, headers: headers, receive_timeout: 60_000) do
      {:ok, %{status: 200, body: body}} ->
        output_file = "data/out/batch_results_#{batch_id}.jsonl"

        # Parse JSONL results
        results =
          String.split(body, "\n")
          |> Enum.filter(&(String.trim(&1) != ""))
          |> Enum.map(fn line ->
            case JSON.decode(line) do
              {:ok, json} -> json
              {:error, _} -> nil
            end
          end)
          |> Enum.filter(&(&1 != nil))

        IO.puts("Processing #{length(results)} successful OpenAI batch results...")
        
        # Count successful results
        successful_count = Enum.count(results, fn r -> 
          get_in(r, ["response", "status_code"]) == 200
        end)
        
        if successful_count < length(results) do
          IO.puts("⚠️  #{length(results) - successful_count} requests had non-200 status codes")
        end

        # Convert OpenAI results to our format
        results
        |> Enum.with_index()
        |> Enum.each(fn {result, index} ->
          if rem(index + 1, 10) == 0 or index + 1 == length(results) do
            IO.puts("Processing result #{index + 1}/#{length(results)}...")
          end

          # Extract text from OpenAI response
          text =
            case get_in(result, ["response", "body", "choices"]) do
              [%{"message" => %{"content" => content}} | _] -> content
              _ -> ""
            end

          if text != "" do
            evaluations = parse_evaluation_response(text)

            formatted_result = %{
              batch_id: result["custom_id"],
              index: index,
              timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
              results: evaluations
            }

            File.write!(output_file, JSON.encode!(formatted_result) <> "\n", [:append])
            IO.puts("Saved result #{index + 1} to file")
          else
            IO.puts("No text found in result #{index + 1}")
            IO.puts("Result structure: #{inspect(result)}")
          end
        end)

        IO.puts("Results saved to: #{output_file}")

      {:ok, %{status: status, body: body}} ->
        IO.puts("Error downloading OpenAI results: HTTP #{status}: #{inspect(body)}")

      {:error, reason} ->
        IO.puts("Failed to download OpenAI results: #{inspect(reason)}")
    end
  end

  defp process_inline_batch_results(responses, batch_id) do
    output_file = "data/out/batch_results_#{String.replace(batch_id, "/", "_")}.jsonl"

    IO.puts("Processing #{length(responses)} responses...")

    responses
    |> Enum.with_index()
    |> Enum.each(fn {response, index} ->
      IO.puts("Processing response #{index + 1}...")

      # Process each response similar to our inline batch processing
      text =
        get_in(response, [
          "response",
          "candidates",
          Access.at(0),
          "content",
          "parts",
          Access.at(0),
          "text"
        ])

      if text do
        evaluations = parse_evaluation_response(text)

        # Get batch metadata if available
        batch_metadata = response["metadata"] || %{}

        result = %{
          batch_id: batch_metadata["batch_id"],
          index: batch_metadata["index"],
          timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
          results: evaluations
        }

        File.write!(output_file, JSON.encode!(result) <> "\n", [:append])
        IO.puts("Saved result #{index + 1} to file")
      else
        IO.puts("No text found in response #{index + 1}")
        IO.puts("Response structure: #{inspect(response)}")
      end
    end)

    IO.puts("Results saved to: #{output_file}")
  end

  defp check_all_batches do
    IO.puts("🔍 Checking all batch jobs...")

    batch_info_files = find_batch_info_files()

    if Enum.empty?(batch_info_files) do
      IO.puts("No batch info files found in #{@output_dir}")
    else
      IO.puts("Found #{length(batch_info_files)} batch info files\n")

      # Load and check each batch
      batch_statuses =
        batch_info_files
        |> Enum.map(&load_and_check_batch/1)
        |> Enum.filter(&(&1 != nil))

      # Group by status
      status_groups = Enum.group_by(batch_statuses, fn batch -> batch.status end)

      # Display summary with all possible statuses
      IO.puts("=== Batch Status Summary ===")

      # Define all possible batch states with descriptions
      all_states = [
        {"BATCH_STATE_PENDING", "🕐 Pending", "Job created and waiting to be processed",
         ["BATCH_STATE_PENDING"]},
        {"BATCH_STATE_RUNNING", "⚙️  Running", "Job is in progress",
         ["BATCH_STATE_RUNNING", "BATCH_STATE_IN_PROGRESS"]},
        {"BATCH_STATE_SUCCEEDED", "✅ Completed", "Job completed successfully",
         ["BATCH_STATE_SUCCEEDED"]},
        {"BATCH_STATE_FAILED", "❌ Failed", "Job failed - check error details",
         ["BATCH_STATE_FAILED"]},
        {"BATCH_STATE_CANCELLED", "🚫 Cancelled", "Job was cancelled by user",
         ["BATCH_STATE_CANCELLED"]},
        {"BATCH_STATE_EXPIRED", "⏰ Expired", "Job expired (>48hrs) - no results available",
         ["BATCH_STATE_EXPIRED"]},
        {"API_ERROR", "🔌 API Error", "Unable to check status via API", ["API_ERROR"]}
      ]

      # Show counts for each state
      for {_state, emoji_status, description, state_variants} <- all_states do
        batches =
          state_variants
          |> Enum.flat_map(fn variant -> Map.get(status_groups, variant, []) end)

        count = length(batches)

        if count > 0 do
          IO.puts("#{emoji_status}: #{count} batches - #{description}")

          for batch <- Enum.take(batches, 3) do
            IO.puts(
              "  • #{batch.batch_id} (#{batch.profiles_to_process} profiles) - #{batch.created_at}"
            )
          end

          if length(batches) > 3 do
            IO.puts("    ... and #{length(batches) - 3} more")
          end

          IO.puts("")
        else
          IO.puts("#{emoji_status}: 0 batches")
        end
      end

      # Show totals
      total_profiles = batch_statuses |> Enum.map(& &1.profiles_to_process) |> Enum.sum()

      completed_profiles =
        status_groups
        |> Map.get("BATCH_STATE_SUCCEEDED", [])
        |> Enum.map(& &1.profiles_to_process)
        |> Enum.sum()

      IO.puts("Total profiles across all batches: #{total_profiles}")
      IO.puts("Completed profiles: #{completed_profiles}")

      if completed_profiles > 0 do
        IO.puts(
          "\nUse 'elixir evaluate_profiles.exs retrieve_batches' to collect all completed results."
        )
      end
    end
  end

  defp retrieve_all_batches do
    IO.puts("📥 Retrieving all completed batch results...")

    batch_info_files = find_batch_info_files()

    if Enum.empty?(batch_info_files) do
      IO.puts("No batch info files found in #{@output_dir}")
    else
      # Load batch info and check status
      completed_batches =
        batch_info_files
        |> Enum.map(&load_and_check_batch/1)
        |> Enum.filter(&(&1 != nil and &1.status == "BATCH_STATE_SUCCEEDED"))

      if Enum.empty?(completed_batches) do
        IO.puts("No completed batches found.")
      else
        IO.puts("Found #{length(completed_batches)} completed batches\n")

        total_retrieved =
          completed_batches
          |> Enum.reduce(0, fn batch, acc ->
            IO.puts("Retrieving #{batch.batch_id}...")

            # The retrieve_batch_results function doesn't return :ok, so let's just call it
            retrieve_batch_results(batch.batch_id)
            IO.puts("✅ Retrieved #{batch.profiles_to_process} profiles")

            acc + batch.profiles_to_process
          end)

        IO.puts("\n=== Batch Retrieval Complete ===")
        IO.puts("Total profiles retrieved: #{total_retrieved}")
      end
    end
  end

  defp find_batch_info_files do
    File.ls!(@output_dir)
    |> Enum.filter(&String.ends_with?(&1, "_batch_info.json"))
    |> Enum.map(&Path.join(@output_dir, &1))
  end

  defp load_and_check_batch(batch_info_file) do
    try do
      batch_info =
        File.read!(batch_info_file)
        |> JSON.decode!()

      batch_id = batch_info["batch_id"]

      # Check current status via API
      api_key = System.get_env("GEMINI_API_KEY")
      if !api_key, do: raise("GEMINI_API_KEY environment variable not set")

      batch_name =
        if String.starts_with?(batch_id, "batches/"), do: batch_id, else: "batches/#{batch_id}"

      url = "https://generativelanguage.googleapis.com/v1beta/#{batch_name}"
      headers = [{"x-goog-api-key", api_key}]

      case Req.get(url, headers: headers, receive_timeout: 30_000) do
        {:ok, %{status: 200, body: body}} ->
          status = body["metadata"]["state"] || body["state"] || "UNKNOWN"

          %{
            batch_id: batch_id,
            status: status,
            profiles_to_process: batch_info["profiles_to_process"] || 0,
            created_at: batch_info["created_at"],
            model: batch_info["model"],
            batch_size: batch_info["batch_size"]
          }

        {:ok, %{status: status, body: body}} ->
          IO.puts("API Error for #{batch_id}: HTTP #{status} - #{inspect(body)}")

          %{
            batch_id: batch_id,
            status: "API_ERROR",
            profiles_to_process: batch_info["profiles_to_process"] || 0,
            created_at: batch_info["created_at"],
            model: batch_info["model"],
            batch_size: batch_info["batch_size"]
          }

        {:error, reason} ->
          IO.puts("Network error for #{batch_id}: #{inspect(reason)}")

          %{
            batch_id: batch_id,
            status: "API_ERROR",
            profiles_to_process: batch_info["profiles_to_process"] || 0,
            created_at: batch_info["created_at"],
            model: batch_info["model"],
            batch_size: batch_info["batch_size"]
          }
      end
    rescue
      # Skip invalid files
      _ -> nil
    end
  end

  defp extract_json_object(text, verbose) do
    # Find the first { and extract a complete JSON object by matching braces
    case :binary.match(text, "{") do
      {start_pos, _} ->
        text_from_brace = String.slice(text, start_pos, String.length(text))

        # Extract preceding text if verbose
        if start_pos > 0 and verbose >= 2 do
          prefix = String.slice(text, 0, start_pos)

          if String.trim(prefix) != "" do
            IO.puts(
              "   Text before JSON: #{String.slice(prefix, 0, min(100, String.length(prefix)))}..."
            )
          end
        end

        # Count braces to find the complete JSON object
        extract_complete_json(text_from_brace)

      :nomatch ->
        nil
    end
  end

  defp extract_complete_json(text) do
    chars = String.graphemes(text)
    extract_json_with_braces(chars, 0, "", false)
  end

  defp extract_json_with_braces([], _depth, acc, _in_string), do: acc

  defp extract_json_with_braces([char | rest], depth, acc, in_string) do
    cond do
      # Handle string boundaries (ignore braces inside strings)
      char == "\"" and not in_string ->
        extract_json_with_braces(rest, depth, acc <> char, true)

      char == "\"" and in_string and String.ends_with?(acc, "\\") == false ->
        extract_json_with_braces(rest, depth, acc <> char, false)

      in_string ->
        extract_json_with_braces(rest, depth, acc <> char, in_string)

      # Count braces when not in string
      char == "{" ->
        extract_json_with_braces(rest, depth + 1, acc <> char, in_string)

      char == "}" ->
        new_acc = acc <> char

        if depth == 1 do
          # Found the closing brace for our JSON object
          new_acc
        else
          extract_json_with_braces(rest, depth - 1, new_acc, in_string)
        end

      true ->
        extract_json_with_braces(rest, depth, acc <> char, in_string)
    end
  end

  defp parse_evaluation_response(response, verbose \\ 0) do
    # First try to extract JSON from the response (Claude sometimes adds explanatory text)
    json_response =
      cond do
        # Check if response starts with JSON
        String.starts_with?(String.trim(response), "{") ->
          response

        # Try to extract JSON block from markdown code block
        String.contains?(response, "```json") ->
          if verbose >= 2 do
            IO.puts("📋 Model provided JSON in markdown code block")
          end

          response
          |> String.split("```json")
          |> Enum.at(1)
          |> then(fn json_part ->
            if json_part do
              json_part
              |> String.split("```")
              |> Enum.at(0)
              |> String.trim()
            else
              nil
            end
          end)

        # Try to find JSON starting with {
        String.contains?(response, "{\n") or String.contains?(response, "{ ") ->
          if verbose >= 2 do
            IO.puts("📋 Model provided JSON with explanatory text")
          end

          # Extract JSON object by finding matching braces
          extract_json_object(response, verbose)

        true ->
          nil
      end

    # Try to parse the extracted JSON
    if json_response do
      case JSON.decode(json_response) do
        {:ok, json} when is_map(json) ->
          # Structured JSON response
          if is_list(json["evaluations"]) do
            json["evaluations"]
            |> Enum.map(fn eval ->
              %{
                "linkedin_id" => eval["linkedin_id"],
                "evaluation" => eval["evaluation"] || eval["decision"]
              }
            end)
          else
            # Fall back to text parsing
            parse_text_evaluation(response)
          end

        _ ->
          # Fall back to text parsing
          parse_text_evaluation(response)
      end
    else
      # Fall back to text parsing
      parse_text_evaluation(response)
    end
  end

  defp parse_text_evaluation(response) do
    response
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.filter(&(&1 != ""))
    |> Enum.map(fn line ->
      case Regex.run(~r/^\[([^\]]+)\]\s*:\s*(ACCEPT|REJECT)\s*$/i, line) do
        [_, linkedin_id, evaluation] ->
          %{"linkedin_id" => String.trim(linkedin_id), "evaluation" => String.upcase(evaluation)}

        _ ->
          nil
      end
    end)
    |> Enum.filter(&(&1 != nil))
  end
end

# Run the script
ProfileEvaluator.main(System.argv())
