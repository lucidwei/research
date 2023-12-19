Set xlApp = CreateObject("Excel.Application")
xlApp.Visible = True ' 或设置为 False 以在后台运行

Set xlBook = xlApp.Workbooks.Open("D:\WPS云盘\WPS云盘\工作-麦高\定期汇报\日报模板整理\融资与北向与全A.xlsx", 0, True)
xlApp.Run "UpdateAndSaveWorkbook"
xlBook.Close False
xlApp.Quit

Set xlBook = Nothing
Set xlApp = Nothing
