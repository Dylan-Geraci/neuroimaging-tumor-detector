# Changelog

## Recent Updates

### Upload Progress Indicator (February 2026)
- Two-phase loading UI: real progress bar during upload, breathing-dot spinner during analysis
- Replaced `fetch()` with `XMLHttpRequest` for real-time upload progress tracking
- Progress bar shows percentage (0–100%) from actual bytes transferred
- Dynamic text adapts to file count ("Uploading 5 scans..." / "Analyzing 5 scans...")
- 400ms analysis phase provides visual continuity for fast uploads

### Batch Processing Support (January 2026)
- Added multi-file upload with drag-and-drop
- Implemented `/predict/batch` API endpoint
- Aggregated diagnosis with agreement scoring
- Expandable individual scan results view
- File preview with thumbnails and management

### Bug Fixes
- Fixed "View Individual Scans" button click handler
- Removed redundant onclick attributes
- Cleaned up debug console.log statements

---

## Next Steps: Frontend Improvements

### High Priority
- ~~Overhauling the look~~ ✓
- ~~Enhanced error handling with detailed inline messages~~ ✓
- ~~Upload progress indicator (% complete if needed)~~ ✓
- ~~Results export (PDF/CSV)~~ ✓

### Medium Priority
- Comparison view for batch predictions
- Keyboard shortcuts for power users
- Accessibility improvements (ARIA labels)

### Future Considerations
- Lazy loading for large batches
- Confidence distribution visualization
- Interactive heatmap overlays
