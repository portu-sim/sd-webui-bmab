from sd_bmab.base import filter


def run_preprocess_filter(context):
	module_config = context.args.get('module_config', {})
	filter_name = module_config.get('preprocess_filter', None)
	if filter_name is None or filter_name == 'None':
		return

	bmab_filter = filter.get_filter(filter_name)
	filter.preprocess_filter(bmab_filter, context, None)
	filter.process_filter(bmab_filter, context, None, None)
	filter.postprocess_filter(bmab_filter, context)
