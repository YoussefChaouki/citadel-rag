#!/bin/bash

##############################################################################
# CITADEL Evaluation Dataset Setup
#
# This script automatically ingests the ML markdown files into CITADEL's
# database for evaluation purposes.
#
# Usage:
#   ./scripts/setup_eval.sh
#
# Environment:
#   API_URL: Base URL of CITADEL API (default: http://localhost:8001)
#   WAIT_TIME: Seconds to wait between ingestions (default: 5)
##############################################################################

set -e  # Exit on error

# Configuration
API_URL="${API_URL:-http://localhost:8001}"
RAG_ENDPOINT="${API_URL}/api/v1/rag"
WAIT_TIME="${WAIT_TIME:-5}"
TEST_DATA_DIR="tests/data/sample_docs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if API is reachable
check_api_health() {
    print_info "Checking API health at ${API_URL}..."

    if ! curl -s -m 5 "${API_URL}/health" > /dev/null 2>&1; then
        print_error "API not reachable at ${API_URL}"
        print_info "Make sure CITADEL is running:"
        print_info "  make up  (Docker mode)"
        print_info "  OR in development mode:"
        print_info "    Terminal 1: make deps"
        print_info "    Terminal 2: make run-citadel"
        print_info "    Terminal 3: make run-ui"
        exit 1
    fi

    print_success "API is healthy"
}

# Check if test data directory exists
check_test_data() {
    print_info "Checking for test data in ${TEST_DATA_DIR}/..."

    if [ ! -d "${TEST_DATA_DIR}" ]; then
        print_error "Directory '${TEST_DATA_DIR}' not found"
        print_info "Please ensure test data files are in: ./${TEST_DATA_DIR}/"
        print_info "Expected files:"
        print_info "  - test_data/ml_fundamentals.md"
        print_info "  - test_data/ml_algorithms.md"
        print_info "  - test_data/ml_deep_learning.md"
        print_info "  - test_data/ml_practical.md"
        exit 1
    fi

    # Check for at least one file
    if ! ls "${TEST_DATA_DIR}"/*.md 1> /dev/null 2>&1; then
        print_error "No .md files found in ${TEST_DATA_DIR}/"
        exit 1
    fi

    print_success "Test data directory found"
}

# Ingest a single file
ingest_file() {
    local file_path=$1
    local file_name=$(basename "$file_path")

    print_info "Ingesting: ${file_name}..."

    # Check if file exists
    if [ ! -f "$file_path" ]; then
        print_error "File not found: ${file_path}"
        return 1
    fi

    # Get file size
    local file_size=$(wc -c < "$file_path")
    local size_mb=$(echo "scale=2; $file_size / 1024 / 1024" | bc)
    print_info "  Size: ${size_mb} MB"

    # Upload file
    local response
    response=$(curl -s -X POST \
        -F "file=@${file_path}" \
        "${RAG_ENDPOINT}/ingest" \
        -m 120)  # 2-minute timeout for large files

    # Check response
    if echo "$response" | grep -q '"status":"success"'; then
        local chunks=$(echo "$response" | grep -o '"chunks_created":[0-9]*' | cut -d: -f2)
        print_success "Ingested '${file_name}' (${chunks} chunks)"
        return 0
    elif echo "$response" | grep -q '"status":"duplicate"'; then
        print_warning "File already exists: ${file_name}"
        return 0
    else
        print_error "Failed to ingest ${file_name}"
        print_info "Response: $response"
        return 1
    fi
}

# List all ingested documents
list_documents() {
    print_info "Retrieving document list..."

    local response
    response=$(curl -s -X GET "${RAG_ENDPOINT}/documents")

    if [ -z "$response" ]; then
        print_error "Failed to retrieve documents"
        return 1
    fi

    # Count documents
    local doc_count=$(echo "$response" | grep -o '"filename"' | wc -l)
    print_success "Found ${doc_count} documents in database"

    if [ "$doc_count" -gt 0 ]; then
        print_info "Documents:"
        echo "$response" | grep -o '"filename":"[^"]*"' | cut -d'"' -f4 | sed 's/^/  - /'
    fi
}

# Main setup flow
main() {
    print_header "CITADEL Evaluation Dataset Setup"

    echo ""
    print_info "This script ingests ML learning materials into CITADEL"
    print_info "for evaluation benchmarking."
    echo ""

    # Checks
    check_api_health
    check_test_data

    echo ""
    print_header "Ingesting Files"

    # Count files to ingest
    local files_to_ingest=()
    while IFS= read -r -d '' file; do
        files_to_ingest+=("$file")
    done < <(find "${TEST_DATA_DIR}" -maxdepth 1 -name "*.md" -print0 | sort -z)

    local total_files=${#files_to_ingest[@]}
    print_info "Will ingest ${total_files} files"
    echo ""

    # Ingest all files
    local failed_files=0
    for ((i=0; i<${#files_to_ingest[@]}; i++)); do
        local file="${files_to_ingest[$i]}"
        local progress=$((i + 1))
        print_info "[$progress/${total_files}]"

        if ! ingest_file "$file"; then
            ((failed_files++))
        fi

        # Wait between files (gives embedding time)
        if [ $i -lt $((total_files - 1)) ]; then
            sleep "$WAIT_TIME"
        fi
    done

    echo ""
    print_header "Ingestion Summary"

    local successful=$((total_files - failed_files))
    print_success "Successfully ingested: ${successful}/${total_files} files"

    if [ $failed_files -gt 0 ]; then
        print_warning "Failed files: ${failed_files}"
    fi

    echo ""
    list_documents

    echo ""
    print_header "Setup Complete"

    if [ $failed_files -eq 0 ]; then
        print_success "All files ingested successfully!"
        print_info "You can now run: make eval"
        echo ""
        print_info "To evaluate CITADEL performance:"
        echo "  make eval"
        echo ""
    else
        print_error "Some files failed to ingest"
        print_info "Check API logs and ensure all files are valid markdown"
        exit 1
    fi
}

# Run main function
main "$@"
