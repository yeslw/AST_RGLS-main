# SPLNPROC Technical Instructions

This repository provides technical instructions and usage notes for the Microsoft Word proceedings paper template used for Springer Computer Science proceedings.

The template is designed to help authors prepare manuscripts in the required Springer proceedings format by using predefined Word styles, custom formatting buttons, and macro-based tools.

## Overview

The main template file is intended for authors preparing papers for Springer proceedings series. It provides predefined formatting styles for common manuscript elements, including:

* Paper title and subtitle
* Author names and ORCID identifiers
* Affiliations, email addresses, and URLs
* Abstract and keywords
* Section and subsection headings
* Lists, footnotes, figures, tables, equations, program code, and references

The template is provided as a macro-enabled Word document. Authors should use the template during manuscript preparation, but the final submission file should be saved as `.docx`.

## Template Files

Typical template files include:

```text
splnproc2510.docm
splnproc2510_mac.docm
```

The Windows version supports Word 2010 and newer. A separate Mac version is provided because Word for Mac may not support all features available in Word for Windows.

## How to Use

### 1. Start from the Template

Open the template file and replace the sample content with your own paper content.

If you have not yet started writing, it is recommended to write directly inside the template.

### 2. Paste Existing Content Carefully

If you copy content from another document into the template, the original formatting may be retained. In that case, select the pasted paragraphs and apply the corresponding Springer styles using the template ribbon.

### 3. Use the Springer Proceedings Macros Ribbon

After enabling macros, Word should display a custom ribbon named:

```text
Springer Proceedings Macros
```

This ribbon provides buttons for formatting the manuscript according to Springer proceedings standards.

Common formatting commands include:

| Element                  | Function                                              |
| ------------------------ | ----------------------------------------------------- |
| Title                    | Formats the contribution title                        |
| Subtitle                 | Formats the contribution subtitle                     |
| Author                   | Formats author names                                  |
| ORCID                    | Formats and checks ORCID identifiers                  |
| Address                  | Formats affiliations and address information          |
| E-mail                   | Formats email addresses and URLs                      |
| Abstract                 | Applies abstract format and adds the “Abstract” label |
| Keywords                 | Applies keyword format and adds the “Keywords” label  |
| H1 / H2 / H3 / H4        | Formats section headings                              |
| Bullet / Dash / Num Item | Creates structured lists                              |
| Footnote                 | Inserts footnotes                                     |
| Reference Item           | Formats reference entries                             |
| Insert Image             | Inserts an image into the manuscript                  |
| Figure Caption           | Formats figure captions                               |
| Table Caption            | Formats table captions                                |
| Displayed Equation       | Formats displayed equations                           |
| Add Eq. Number           | Adds equation numbering                               |
| Prog. Code               | Formats code blocks                                   |
| Restore Styles           | Restores predefined template styles                   |

## Manuscript Formatting Notes

### Headings

Use the heading buttons provided in the ribbon:

```text
H1 -> Level 1 heading
H2 -> Level 2 heading
H3 -> Bold run-in heading
H4 -> Italic run-in heading
```

If a heading should not be numbered, apply the heading style first and then remove the generated number manually.

### Lists

The template provides predefined list styles for:

* Bullet lists
* Dash lists
* Numbered lists
* Nested lists

For nested lists, apply the list style first, then use the list level controls to adjust indentation.

### Figures and Tables

Images should be inserted through the template’s image insertion command when possible.

After inserting an image or table, use the corresponding caption button:

```text
Figure Caption
Table Caption
```

The template automatically adds labels such as:

```text
Fig. X
Table X
```

### Equations

Displayed equations should be formatted with the equation button. Equation numbers can be added only after the displayed equation style has been applied.

### Program Code

Use the program code formatting button for code blocks or command sequences. The predefined formatting includes tab settings for code indentation.

## Saving and Submission

After completing the manuscript, save the final file as:

```text
.docx
```

Do not submit the macro-enabled `.docm` file.

Recommended naming format:

```text
AuthorName_ShortTitle.docx
```

Example:

```text
Smith_TitleOfPaper.docx
```

## Common Issues

### The Springer Proceedings Macros ribbon does not appear

This is usually caused by Word macro security settings. Enable macros in Microsoft Word and reopen the template.

### A formatting button throws an error

This usually means that the predefined styles are missing or have been modified. Click the `Restore Styles` button to restore the template styles.

### Pasted text does not match the Springer format

Copied content may retain its original formatting. Select the affected text, click `Normal Text` twice to reset formatting, and then apply the required Springer style.

### Section numbering is incorrect

Reapply the `H1` or `H2` style to the affected headings to restore the correct numbering.

### Special characters disappear after formatting

Insert special characters through Word’s symbol insertion menu instead of typing them directly when possible.

## Recommended Workflow

1. Open the Springer proceedings template.
2. Enable macros if required.
3. Replace the sample content with manuscript content.
4. Apply the correct style to each manuscript element.
5. Insert figures, tables, equations, and references using the template commands.
6. Use `Restore Styles` if formatting problems occur.
7. Save the final manuscript as `.docx`.
8. Submit the `.docx` file, not the `.docm` template file.

## Notes

This repository is intended as a technical usage guide for the Springer proceedings Word template. It does not replace the official Springer author guidelines. Authors should still check the latest official Springer instructions before submission.
