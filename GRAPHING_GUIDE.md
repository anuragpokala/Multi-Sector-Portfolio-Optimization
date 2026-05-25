# Graphing Guide for Portfolio Backtest Results

This guide provides step-by-step instructions for creating Virginia Tech themed graphs from the Excel output file.

## File Location

After running the backtest script, you'll find the Excel file at:
```
outputs/consumers_optimization.xlsx
```

This file contains 3 tabs with pre-formatted data ready for graphing.

---

## Tab 1: BL/MV Comparison

**Purpose:** Compare Before, Mean-Variance, and Black-Litterman strategies

### Steps to Create Graph:

1. **Open Excel file** and go to the `BL_MV_Comparison` tab

2. **Select data range:**
   - Click on cell A1 (Date header)
   - Hold Shift and click on the last cell in column D
   - This selects: Date, Before, Mean-Variance, Black-Litterman

3. **Insert chart:**
   - Go to Insert → Charts → Line Chart
   - Choose "Line" or "Line with Markers"

4. **Format chart:**
   - Add chart title: "Portfolio Performance: BL vs MV"
   - Add axis titles:
     - X-axis: "Date"
     - Y-axis: "Cumulative Return (%)"

5. **Apply Virginia Tech colors:**
   - Click on the "Before" line → Right-click → Format Data Series
     - Line Color: **Maroon** - RGB(134, 31, 65) or #861F41
   
   - Click on the "Mean-Variance" line → Right-click → Format Data Series
     - Line Color: **Orange** - RGB(232, 119, 34) or #E87722
   
   - Click on the "Black-Litterman" line → Right-click → Format Data Series
     - Line Color: **Red** - RGB(207, 69, 32) or #CF4520

---

## Tab 2: RP/HRP Comparison

**Purpose:** Compare Before, Risk Parity, and HRP strategies

### Steps to Create Graph:

1. **Open Excel file** and go to the `RP_HRP_Comparison` tab

2. **Select data range:**
   - Click on cell A1 (Date header)
   - Hold Shift and click on the last cell in column D
   - This selects: Date, Before, Risk Parity, HRP

3. **Insert chart:**
   - Go to Insert → Charts → Line Chart
   - Choose "Line" or "Line with Markers"

4. **Format chart:**
   - Add chart title: "Portfolio Performance: RP vs HRP"
   - Add axis titles:
     - X-axis: "Date"
     - Y-axis: "Cumulative Return (%)"

5. **Apply Virginia Tech colors:**
   - Click on the "Before" line → Right-click → Format Data Series
     - Line Color: **Maroon** - RGB(134, 31, 65) or #861F41
   
   - Click on the "Risk Parity" line → Right-click → Format Data Series
     - Line Color: **Brown** - RGB(99, 0, 49) or #630031
   
   - Click on the "HRP" line → Right-click → Format Data Series
     - Line Color: **Black** - RGB(0, 0, 0) or #000000

---

## Tab 3: Hybrid Comparison

**Purpose:** Compare Before portfolio with Hybrid BL-HRP strategy

### Steps to Create Graph:

1. **Open Excel file** and go to the `Hybrid_Comparison` tab

2. **Select data range:**
   - Click on cell A1 (Date header)
   - Hold Shift and click on the last cell in column C
   - This selects: Date, Before, Hybrid

3. **Insert chart:**
   - Go to Insert → Charts → Line Chart
   - Choose "Line" or "Line with Markers"

4. **Format chart:**
   - Add chart title: "Portfolio Performance: Hybrid BL-HRP"
   - Add axis titles:
     - X-axis: "Date"
     - Y-axis: "Cumulative Return (%)"

5. **Apply Virginia Tech colors:**
   - Click on the "Before" line → Right-click → Format Data Series
     - Line Color: **Maroon** - RGB(134, 31, 65) or #861F41
   
   - Click on the "Hybrid" line → Right-click → Format Data Series
     - Line Color: **Red** - RGB(207, 69, 32) or #CF4520

---

## Virginia Tech Color Palette

### Official Colors

| Color Name | Hex Code | RGB Values |
|------------|----------|------------|
| **Chicago Maroon** | #630031 | RGB(99, 0, 49) |
| **Burnt Orange** | #CF4520 | RGB(207, 69, 32) |
| **Hokie Stone** | #75787B | RGB(117, 120, 123) |

### Extended Palette

| Color Name | Hex Code | RGB Values | Usage |
|------------|----------|------------|-------|
| **Maroon** | #861F41 | RGB(134, 31, 65) | Before portfolio |
| **Brown** | #630031 | RGB(99, 0, 49) | Risk Parity |
| **Orange** | #E87722 | RGB(232, 119, 34) | Mean-Variance |
| **Red** | #CF4520 | RGB(207, 69, 32) | Black-Litterman, Hybrid |
| **Black** | #000000 | RGB(0, 0, 0) | HRP |

---

## Tips for Better Graphs

### Line Styling

1. **Line width:** Set to 2-3 pt for better visibility
   - Right-click line → Format Data Series → Line → Width

2. **Add markers:** Helps distinguish lines at data points
   - Format Data Series → Marker → Built-in → Size: 5-7

3. **Legend position:** Place legend on the right or bottom
   - Click legend → Drag to desired position
   - Or: Right-click → Format Legend → Position

### Chart Formatting

1. **Remove gridlines** for cleaner look:
   - Click on gridlines → Delete
   - Or: Chart Design → Add Chart Element → Gridlines → None

2. **Format axes:**
   - Date axis: Auto format usually works well
   - Y-axis (Returns): Set min to reasonable value (e.g., -20% or 0%)

3. **Add data labels** (optional):
   - Select a line → Right-click → Add Data Labels
   - Format to show only at specific points if desired

---

## Quick Reference: Selecting Data

### Method 1: Click and Drag
- Click on A1
- Hold mouse button and drag to last cell in the range
- Release

### Method 2: Keyboard Shortcut
- Click on A1
- Hold Shift
- Press Ctrl+↓ to select to bottom of data
- Press Ctrl+→ to extend selection to last column
- Release

### Method 3: Name Box
- Click Name Box (left of formula bar)
- Type range (e.g., `A1:D755`)
- Press Enter

---

## Troubleshooting

### Issue: Chart doesn't show all data
**Solution:** Make sure you selected all rows including the header row (row 1)

### Issue: Date axis looks wrong
**Solution:** 
- Right-click X-axis → Format Axis
- Ensure "Date axis" is selected (not "Text axis")

### Issue: Lines are hard to distinguish
**Solution:**
- Increase line width (2-3 pt)
- Add markers to each line
- Use contrasting colors from the Virginia Tech palette

### Issue: Can't find RGB color input
**Solution:**
- In "Format Data Series" → Line → Color → More Colors
- Click "Custom" tab at the top
- Enter RGB values or Hex code

---

## Example Output

Your final graphs should look similar to the cumulative returns plot shown in the notebook, with:
- Clean lines in Virginia Tech colors
- Clear legend identifying each strategy
- Appropriate axis labels
- Professional appearance

For questions or issues, refer to the main README.md or contact the project maintainer.
